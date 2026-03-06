import logging
import sys
from pathlib import Path


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Creates a structured logger with console and file handlers.
    
    Usage:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Starting service...")
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger  # Avoid duplicate handlers
    
    logger.setLevel(level)
    
    # Console handler with colored-like formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)
    
    # File handler
    from logging.handlers import TimedRotatingFileHandler
    log_dir = Path(__file__).resolve().parent.parent.parent / "data"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = TimedRotatingFileHandler(
        log_dir / "system.log",
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)
    
    return logger
