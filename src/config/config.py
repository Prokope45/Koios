import os
from typing import List
from dotenv import load_dotenv
from pathlib import Path


class Config:
    """Singleton config shared across the app."""

    _instance = None

    def __new__(cls):
        """Create instance if not yet created. Otherwise return instance."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the config instance if not already initialized."""
        if self._initialized:
            return
        self.setup()
        self._initialized = True

    def setup(self) -> None:
        """Load environment variables from .env file."""
        path: str = Path("./.env")
        load_dotenv(path)

    @property
    def enable_internet_search(self) -> bool:
        return os.getenv("ENABLE_INTERNET_SEARCH", "False").lower() == "true"

    @property
    def chat_history_db_path(self) -> str:
        """Filesystem path to the SQLite chat-history database.

        Defaults to `db/chat_history.sqlite` which is the same directory
        used by ChromaDB so that the existing Docker volume mount covers it.
        """
        return os.getenv("CHAT_HISTORY_DB_PATH", "db/chat_history.sqlite")

    @property
    def approved_user_ids(self) -> List[str]:
        """Return the list of approved API user identifiers.

        Read from the `APPROVED_USER_IDS` environment variable as a
        comma-separated string, e.g.::

            APPROVED_USER_IDS=alice,bob,charlie

        Returns an empty list if the variable is not set, which the API server
        treats as *no users approved* (all requests rejected).
        """
        raw = os.getenv("APPROVED_USER_IDS", "")
        return [uid.strip() for uid in raw.split(",") if uid.strip()]

    @property
    def max_messages_per_user(self) -> int:
        """Return the max number of messages for each user.

        Read from the `MAX_MESSAGES_PER_USER` environment variable.

        Returns 500 by default if the variable is not set.
        """
        return int(os.getenv("MAX_MESSAGES_PER_USER", 500))
