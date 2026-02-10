import os
from dotenv import load_dotenv
from pathlib import Path

class Config:

    def setup(self):
        path: str = Path("src/.env")
        load_dotenv(path)
