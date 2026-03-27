"""
CAP Logging Setup
=================
Configures Python's logging module for the entire application.
Each layer gets its own named logger under the 'cap' namespace.
Output goes to both console and a rotating file.

Usage from any module:
    import logging
    logger = logging.getLogger("cap.layer2.capture")
    logger.info("Scan started")

The audit_log database table (Layer 5) is separate — it records
user-facing events. This module handles developer diagnostics only.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional


# Pre-defined layer loggers for convenience.
# Any module can also create sub-loggers like "cap.layer2.capture".
LAYER_LOGGERS = [
    "cap.motor",
    "cap.safety",
    "cap.scan_region",
    "cap.oil",
    "cap.focus",
    "cap.camera",
    "cap.capture",
    "cap.stacker",
    "cap.processing",
    "cap.inference",
    "cap.data",
    "cap.ui",
    "cap.visualization",
    "cap.metrics",
    "cap.retraining",
    "cap.worker",
    "cap.config",
    "cap.app",
]

LOG_FORMAT = "%(asctime)s | %(name)-24s | %(levelname)-8s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    log_dir: str = "./logs",
    console_level: str = "DEBUG",
    file_level: str = "DEBUG",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    log_filename: str = "cap.log",
) -> logging.Logger:
    """
    Configure the root 'cap' logger with console and file handlers.

    Parameters
    ----------
    log_dir : str
        Directory for log files. Created if it doesn't exist.
    console_level : str
        Log level for console output ("DEBUG", "INFO", "WARNING", etc.).
    file_level : str
        Log level for file output. Always DEBUG in development.
    max_bytes : int
        Maximum size per log file before rotation (default 10 MB).
    backup_count : int
        Number of rotated log files to keep (default 5).
    log_filename : str
        Name of the log file within log_dir.

    Returns
    -------
    logging.Logger
        The root 'cap' logger, fully configured.
    """
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Get or create the root CAP logger
    root_logger = logging.getLogger("cap")
    root_logger.setLevel(logging.DEBUG)  # Allow all levels through; handlers filter

    # Avoid duplicate handlers if configure_logging is called multiple times
    if root_logger.handlers:
        root_logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, console_level.upper(), logging.DEBUG))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Rotating file handler
    log_path = os.path.join(log_dir, log_filename)
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    root_logger.info(
        "Logging configured: console=%s, file=%s (%s)",
        console_level.upper(),
        file_level.upper(),
        log_path,
    )

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Convenience function to get a named logger under the 'cap' namespace.

    If the name doesn't start with 'cap.', it is prefixed automatically.

    Parameters
    ----------
    name : str
        Logger name, e.g. "layer2.capture" → becomes "cap.layer2.capture".

    Returns
    -------
    logging.Logger
    """
    if not name.startswith("cap."):
        name = f"cap.{name}"
    return logging.getLogger(name)
