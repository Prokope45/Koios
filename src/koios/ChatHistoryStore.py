"""ChatHistoryStore.py

SQLite-backed chat history store for per-user conversation persistence.

Each user's messages are stored in a single ``chat_messages`` table keyed by
``user_id``.  A sliding-window cap of ``MAX_MESSAGES_PER_USER`` (100) is
enforced so that the table never grows unbounded.

Author: Jared Paubel jpaubel@pm.me
version 0.1.0
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import List

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
    select,
    delete,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import logger

# Maximum number of messages retained per user (sliding window).
MAX_MESSAGES_PER_USER: int = 100


class _Base(DeclarativeBase):
    pass


class ChatMessageRecord(_Base):
    """ORM model for a single chat message belonging to a user.

    Attributes:
        id (int): Auto-incrementing primary key.
        user_id (str): Opaque user identifier supplied via ``X-User-ID`` header.
        role (str): Either ``"user"`` or ``"assistant"``.
        content (str): The message text.
        created_at (datetime): UTC timestamp of insertion.
    """

    __tablename__ = "chat_messages"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    user_id: str = Column(String(256), nullable=False, index=True)
    role: str = Column(String(16), nullable=False)
    content: str = Column(Text, nullable=False)
    created_at: datetime = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self) -> dict:
        """Return a plain ``{"role", "content"}`` dict for use with the agent."""
        return {"role": self.role, "content": self.content}


class ChatHistoryStore:
    """Persistent, per-user chat history backed by SQLite.

    Usage::

        store = ChatHistoryStore()
        store.add_message("alice", "user", "Hello!")
        history = store.get_history("alice")  # [{"role": "user", "content": "Hello!"}]

    Args:
        db_path (str): Filesystem path to the SQLite database file.
            Defaults to ``db/chat_history.sqlite``.
    """

    def __init__(self, db_path: str = "db/chat_history.sqlite") -> None:
        # Ensure the parent directory exists (mirrors how ChromaDB uses db/).
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
            echo=False,
        )
        _Base.metadata.create_all(engine)
        self._Session: sessionmaker[Session] = sessionmaker(bind=engine)
        logger.info("ChatHistoryStore initialised at %s", db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_history(self, user_id: str) -> List[dict]:
        """Return the stored chat history for *user_id* as a list of dicts.

        Messages are returned in chronological order (oldest first) and are
        capped at :data:`MAX_MESSAGES_PER_USER` entries.

        Args:
            user_id (str): The user identifier.

        Returns:
            list[dict]: List of ``{"role": ..., "content": ...}`` dicts.
        """
        with self._Session() as session:
            stmt = (
                select(ChatMessageRecord)
                .where(ChatMessageRecord.user_id == user_id)
                .order_by(ChatMessageRecord.created_at.asc())
                .limit(MAX_MESSAGES_PER_USER)
            )
            records = session.scalars(stmt).all()
            return [r.to_dict() for r in records]

    def add_message(self, user_id: str, role: str, content: str) -> None:
        """Append a single message to *user_id*'s history.

        If the user already has :data:`MAX_MESSAGES_PER_USER` messages stored,
        the oldest message is deleted before the new one is inserted (sliding
        window).

        Args:
            user_id (str): The user identifier.
            role (str): ``"user"`` or ``"assistant"``.
            content (str): The message text.
        """
        with self._Session() as session:
            # Count existing messages for this user.
            count: int = session.scalar(
                select(func.count()).where(ChatMessageRecord.user_id == user_id)
            ) or 0

            if count >= MAX_MESSAGES_PER_USER:
                # Delete the oldest message(s) to make room.
                overflow = count - MAX_MESSAGES_PER_USER + 1
                oldest_ids_stmt = (
                    select(ChatMessageRecord.id)
                    .where(ChatMessageRecord.user_id == user_id)
                    .order_by(ChatMessageRecord.created_at.asc())
                    .limit(overflow)
                )
                oldest_ids = list(session.scalars(oldest_ids_stmt).all())
                session.execute(
                    delete(ChatMessageRecord).where(
                        ChatMessageRecord.id.in_(oldest_ids)
                    )
                )

            session.add(
                ChatMessageRecord(
                    user_id=user_id,
                    role=role,
                    content=content,
                    created_at=datetime.now(timezone.utc),
                )
            )
            session.commit()

    def add_messages(self, user_id: str, messages: List[dict]) -> None:
        """Convenience wrapper to append multiple messages at once.

        Args:
            user_id (str): The user identifier.
            messages (list[dict]): List of ``{"role", "content"}`` dicts.
        """
        for msg in messages:
            self.add_message(user_id, msg["role"], msg["content"])

    def clear_history(self, user_id: str) -> int:
        """Delete all stored messages for *user_id*.

        Args:
            user_id (str): The user identifier.

        Returns:
            int: Number of messages deleted.
        """
        with self._Session() as session:
            result = session.execute(
                delete(ChatMessageRecord).where(
                    ChatMessageRecord.user_id == user_id
                )
            )
            session.commit()
            deleted = result.rowcount
            logger.info(
                "Cleared %d message(s) for user '%s'", deleted, user_id
            )
            return deleted

    def get_message_count(self, user_id: str) -> int:
        """Return the number of messages currently stored for *user_id*.

        Args:
            user_id (str): The user identifier.

        Returns:
            int: Message count.
        """
        with self._Session() as session:
            return session.scalar(
                select(func.count()).where(ChatMessageRecord.user_id == user_id)
            ) or 0

    def list_users(self) -> List[str]:
        """Return a list of all user IDs that have stored messages.

        Returns:
            list[str]: Distinct user IDs.
        """
        with self._Session() as session:
            rows = session.scalars(
                select(ChatMessageRecord.user_id).distinct()
            ).all()
            return list(rows)
