"""
logger.py — Centralized logger factory for the flight delay pipeline.

All modules obtain their logger via get_logger(__name__, logging_config).
This ensures consistent formatting, log levels, and optional file output
across the entire pipeline without duplicating handler setup.
"""

import logging
import sys
from pathlib import Path
from typing import Any


def get_logger(name: str, logging_config: dict[str, Any]) -> logging.Logger:
    """
    Build and return a configured logger.

    Args:
        name:           Module __name__ — becomes the logger hierarchy name.
        logging_config: The 'logging' section from pipeline.yaml.

    Returns:
        Configured logging.Logger instance.

    Note:
        Calling this multiple times with the same name is safe — Python's
        logging module returns the same logger instance and handlers are
        checked for duplicates before adding.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    level_str = logging_config.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)

    fmt = logging_config.get(
        "format", "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    datefmt = logging_config.get("datefmt", "%Y-%m-%d %H:%M:%S")
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Console handler — always added
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler — added only when log_to_file is True
    if logging_config.get("log_to_file", False):
        log_file = logging_config.get("log_file", "pipeline.log")
        _add_file_handler(logger, log_file, formatter, level)

    # Prevent propagation to root logger to avoid duplicate output
    logger.propagate = False

    return logger


def _add_file_handler(
    logger: logging.Logger,
    log_file: str,
    formatter: logging.Formatter,
    level: int,
) -> None:
    """
    Attach a FileHandler to the logger.

    Creates parent directories if they don't exist. Appends to the log
    file rather than overwriting, so multiple pipeline runs accumulate.
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        logger.warning(
            "Could not create log file at '%s': %s. Logging to console only.",
            log_file,
            exc,
        )
