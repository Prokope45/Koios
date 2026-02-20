import os
import logging
from typing import List, Optional
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

    @property
    def jwt_secret_key(self) -> str:
        """Secret key used to verify incoming JWT tokens.

        Read from the ``KOIOS_JWT_SECRET_KEY`` environment variable.  This
        value **must** be set; the API server will reject all authenticated
        requests if it is absent.
        """
        return os.getenv("KOIOS_JWT_SECRET_KEY", "")

    @property
    def jwt_algorithm(self) -> str:
        """Algorithm used to decode JWT tokens.

        Defaults to ``HS256``.  Override via ``KOIOS_JWT_ALGORITHM``.
        """
        return os.getenv("KOIOS_JWT_ALGORITHM", "HS256")

    @property
    def jwt_expiry_hours(self) -> Optional[int]:
        """Optional token expiry window in hours.

        When set, the API server will reject tokens whose ``exp`` claim
        indicates they have expired.  When absent (or empty), expiry is not
        enforced.

        Read from ``KOIOS_JWT_EXPIRY_HOURS``.
        """
        raw = os.getenv("KOIOS_JWT_EXPIRY_HOURS", "")
        if raw.strip():
            try:
                return int(raw.strip())
            except ValueError:
                return None
        return None

    @property
    def jwt_issuer(self) -> str:
        """Issuer claim (``iss``) embedded in generated JWT tokens.

        Defaults to ``"koios-api"``.  Override via ``KOIOS_JWT_ISSUER``.
        """
        return os.getenv("KOIOS_JWT_ISSUER", "koios-api")

    @property
    def authorized_token_ips(self) -> List[str]:
        """IP addresses permitted to call the ``/token`` endpoint.

        Always includes ``127.0.0.1`` and ``::1`` so that local development
        works without any additional configuration.

        Additional addresses are read from the ``KOIOS_AUTHORIZED_TOKEN_IPS``
        environment variable as a comma-separated string, e.g.::

            KOIOS_AUTHORIZED_TOKEN_IPS=203.0.113.10,198.51.100.42

        Returns:
            List[str]: Deduplicated list of authorised IP addresses.
        """
        # Localhost is always allowed for local development.
        localhost = {"127.0.0.1", "::1"}
        raw = os.getenv("KOIOS_AUTHORIZED_TOKEN_IPS", "")
        configured = {ip.strip() for ip in raw.split(",") if ip.strip()}
        return list(localhost | configured)
