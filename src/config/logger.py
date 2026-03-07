import logging


class KoiosLogger:
    """Logging singleton."""

    _instance = None
    
    def __new__(cls, name: str = "koios"):
        """Create instance if not yet created. Otherwise return instance."""
        if cls._instance is None:
            cls._instance = super(KoiosLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, name: str = "koios"):
        if self._initialized:
            return
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._logger = logging.getLogger(name)
        self._initialized = True

    @property
    def logger(self) -> logging.Logger:
        """Logger getter."""
        return self._logger
