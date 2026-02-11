import os
from dotenv import load_dotenv
from pathlib import Path

class Config:

    def setup(self):
        path: str = Path("src/.env")
        load_dotenv(path)

    @property
    def enable_internet_search(self) -> bool:
        return os.getenv("KOIOS_ENABLE_INTERNET_SEARCH", "False").lower() == "true"
