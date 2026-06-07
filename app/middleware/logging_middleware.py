"""
Request logging middleware.

Attaches a UUID request-id to every request and logs method, URL, client IP,
headers, query params, and body on arrival — formatted as indented JSON.

For multipart/form-data bodies, file fields are logged as their filename
(or "unknown") — raw bytes are never logged.
Response logging is handled by send_response (app/utils/response.py).
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from starlette.datastructures import UploadFile
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from app.core.logging import get_logger

logger = get_logger(__name__)


def _dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception:
        return str(obj)


class LoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        body_log = await _extract_request_body(request)

        logger.info(
            "Request received\n%s",
            _dumps({
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_ip": request.client.host if request.client else "unknown",
                "query_params": dict(request.query_params),
                "headers": {
                    "content-type": request.headers.get("content-type"),
                    "origin": request.headers.get("origin"),
                    "sec-fetch-site": request.headers.get("sec-fetch-site"),
                    "sec-fetch-mode": request.headers.get("sec-fetch-mode"),
                    "sec-ch-ua-platform": request.headers.get("sec-ch-ua-platform"),
                },
                "body": body_log,
            }),
        )

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


async def _extract_request_body(request: Request) -> Any:
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        form = await request.form()
        result: dict[str, Any] = {}
        for key, value in form.multi_items():
            if isinstance(value, UploadFile):
                result[key] = value.filename or "unknown"
            else:
                result[key] = value
        return result

    if "application/json" in content_type:
        try:
            raw = await request.body()
            return json.loads(raw)
        except Exception:
            return "<unparseable JSON>"

    if "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        return dict(form)

    return None
