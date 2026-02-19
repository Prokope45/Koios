import os
import logging
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
