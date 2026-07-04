"""
Centralised logging configuration.

Call `setup_logging()` once at application startup.
All modules should obtain a logger via `get_logger(__name__)`.
"""

from __future__ import annotations

import contextvars
import logging
import sys
from typing import Optional


#: Sentinel for a log record emitted outside any request (CLI, startup,
#: background work); the formatter omits the id segment entirely in that case.
_NO_CONTEXT = "-"

#: Request-scoped ids, populated by LoggingMiddleware at the start of each
#: request and reset when it ends. They live here — framework-free stdlib
#: contextvars — so the pure service layer's logs can carry the ids without
#: importing anything HTTP-specific.
correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=_NO_CONTEXT
)
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=_NO_CONTEXT
)

_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(context)s%(message)s"
_DEFAULT_LEVEL = logging.INFO
_QUIET_LIBRARIES = (
    "httpx",
    "httpcore",
    "huggingface_hub",
)


# The stock factory, captured once so re-running setup_logging stays idempotent.
_BASE_RECORD_FACTORY = logging.getLogRecordFactory()


def _record_factory(*args: object, **kwargs: object) -> logging.LogRecord:
    """Stamp every log record with the active request/correlation ids.

    Runs at record-creation time — in the context of whatever called the logger —
    so a record made while handling a request (including deep in the framework-free
    service layer, e.g. "Model '…' loaded successfully.") captures that request's
    ids, while one made outside a request captures the '-' default. Recording them
    as record attributes makes them available to the formatter's prefix and to
    test inspection (caplog) alike.
    """
    record = _BASE_RECORD_FACTORY(*args, **kwargs)  # type: ignore[arg-type]
    record.correlation_id = correlation_id_var.get()
    record.request_id = request_id_var.get()
    return record


class _ContextFormatter(logging.Formatter):
    """Prefix each record with its request/correlation ids when present.

    Reads the ids stamped on the record by ``_record_factory``. During a request —
    including logs emitted deep in the framework-free service layer (e.g. "Model
    '…' loaded successfully.") — the segment ``[correlation_id=… request_id=…]`` is
    prepended so the whole flow is traceable by id. Outside a request both ids are
    ``-`` and the segment is omitted, keeping CLI and startup logs clean.
    """

    def format(self, record: logging.LogRecord) -> str:
        correlation_id = getattr(record, "correlation_id", _NO_CONTEXT)
        request_id = getattr(record, "request_id", _NO_CONTEXT)
        if correlation_id == _NO_CONTEXT and request_id == _NO_CONTEXT:
            record.context = ""
        else:
            record.context = f"[correlation_id={correlation_id} request_id={request_id}] "
        return super().format(record)


def setup_logging(
    level: int = _DEFAULT_LEVEL,
    fmt: str = _DEFAULT_FORMAT,
) -> None:
    """Configure the root logger with a stream handler to stdout."""
    logging.setLogRecordFactory(_record_factory)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ContextFormatter(fmt))
    logging.basicConfig(
        level=level,
        handlers=[handler],
        force=True,
    )

    # Keep third-party network chatter from breaking progress bars during
    # long-running downloads while preserving normal app-level INFO logs.
    for logger_name in _QUIET_LIBRARIES:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a named logger (or the root logger if name is None)."""
    return logging.getLogger(name)
