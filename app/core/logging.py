"""
Centralised logging configuration.

Call `setup_logging()` once at application startup.
All modules should obtain a logger via `get_logger(__name__)`.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
_DEFAULT_LEVEL = logging.INFO
_QUIET_LIBRARIES = (
    "httpx",
    "httpcore",
    "huggingface_hub",
)


def setup_logging(
    level: int = _DEFAULT_LEVEL,
    fmt: str = _DEFAULT_FORMAT,
) -> None:
    """Configure the root logger with a stream handler to stdout."""
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Keep third-party network chatter from breaking progress bars during
    # long-running downloads while preserving normal app-level INFO logs.
    for logger_name in _QUIET_LIBRARIES:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a named logger (or the root logger if name is None)."""
    return logging.getLogger(name)
