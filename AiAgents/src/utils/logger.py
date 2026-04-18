"""
Structured logging for AiAgents using Python's standard logging module.
Provides JSON-friendly formatting and consistent log levels.
"""

import logging
import sys
from functools import lru_cache


class ColorFormatter(logging.Formatter):
    """Adds ANSI color codes to console log output."""

    COLORS = {
        logging.DEBUG: "\033[36m",    # Cyan
        logging.INFO: "\033[32m",     # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",    # Red
        logging.CRITICAL: "\033[35m", # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def _build_handler() -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        ColorFormatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    return handler


@lru_cache(maxsize=None)
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named, cached logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(_build_handler())
    logger.setLevel(level)
    logger.propagate = False
    return logger
