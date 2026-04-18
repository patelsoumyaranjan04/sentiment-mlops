"""
Centralized logger configuration using loguru.
Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Hello")
"""

import sys
from loguru import logger as _logger
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def get_logger(name: str = "sentiment-mlops"):
    """Return a configured loguru logger."""
    _logger.remove()  # Remove default handler

    # Console handler
    _logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True,
    )

    # File handler — rotates daily
    _logger.add(
        LOG_DIR / "app_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} - {message}",
        level="DEBUG",
    )

    return _logger.bind(name=name)
