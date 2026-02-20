import os
import logging
from typing import List
from dotenv import load_dotenv
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared application logger
# ---------------------------------------------------------------------------
# All modules should import this logger rather than creating their own so that
# log level and handler configuration is controlled from a single place.
#
# Usage in other modules:
#   from src.config import logger
#
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("koios")


class Config:

    def setup(self):
        path: str = Path("src/.env")
        load_dotenv(path)

    @property
    def enable_internet_search(self) -> bool:
        return os.getenv("KOIOS_ENABLE_INTERNET_SEARCH", "False").lower() == "true"

    @property
    def chat_history_db_path(self) -> str:
        """Filesystem path to the SQLite chat-history database.

        Defaults to ``db/chat_history.sqlite`` which is the same directory
        used by ChromaDB so that the existing Docker volume mount covers it.
        """
        return os.getenv("KOIOS_CHAT_HISTORY_DB_PATH", "db/chat_history.sqlite")

    @property
    def approved_user_ids(self) -> List[str]:
        """Return the list of approved API user identifiers.

        Read from the ``KOIOS_APPROVED_USER_IDS`` environment variable as a
        comma-separated string, e.g.::

            KOIOS_APPROVED_USER_IDS=alice,bob,charlie

        Returns an empty list if the variable is not set, which the API server
        treats as *no users approved* (all requests rejected).
        """
        raw = os.getenv("KOIOS_APPROVED_USER_IDS", "")
        return [uid.strip() for uid in raw.split(",") if uid.strip()]
