"""config package level init file.

Author: Jared Paubel jpaubel@pm.me
Version: 0.1.0
"""
from src.config.config import Config
from src.config.logger import KoiosLogger

# Shared config and logger singletons
config = Config()
logger = KoiosLogger().logger

__all__ = ["config", "logger"]
