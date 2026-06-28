"""
send_response utility.

Logs the response body as indented JSON then returns a JSONResponse.
Accepts plain dicts, Pydantic models, or any JSON-serialisable value.

Usage:
    from app.utils.response import send_response

    @router.get("/example")
    async def example(request: Request):
        return send_response(request, {"hello": "world"})
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from app.core.logging import get_logger

logger = get_logger(__name__)


def _dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception:
        return str(obj)


def get_request_id(request: Any) -> str:
    """Return the request_id stashed on request.state, or 'unknown'."""
    return getattr(getattr(request, "state", None), "request_id", "unknown")


def get_correlation_id(request: Any) -> str:
    """Return the correlation_id stashed on request.state, or 'unknown'."""
    return getattr(getattr(request, "state", None), "correlation_id", "unknown")


def log_response(
    request: Any,
    body: Any,
    status_code: int = 200,
) -> Any:
    """Log body as pretty JSON. Returns the JSON-encoded body.

    Use directly for responses that aren't a JSONResponse (e.g. the accumulated
    SSE body of a stream); send_response wraps this for plain JSON replies.
    """
    request_id = get_request_id(request)
    correlation_id = get_correlation_id(request)
    serializable = jsonable_encoder(body)

    envelope = {"correlation_id": correlation_id, "request_id": request_id, "status_code": status_code}
    if isinstance(serializable, str):
        # Raw text body (e.g. the accumulated SSE stream) — print it below the
        # envelope with real newlines instead of burying it as an escaped
        # ("\n\n"-laden) JSON string, which is unreadable for a full stream.
        logger.info("Response sent\n%s\n%s", _dumps(envelope), serializable)
    else:
        envelope["body"] = serializable
        logger.info("Response sent\n%s", _dumps(envelope))

    return serializable


def send_response(
    request: Request,
    body: Any,
    status_code: int = 200,
) -> JSONResponse:
    """Log body as pretty JSON and return a JSONResponse."""
    serializable = log_response(request, body, status_code)
    return JSONResponse(content=serializable, status_code=status_code)
