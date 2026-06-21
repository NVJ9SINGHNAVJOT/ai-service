"""
Request logging middleware.

Attaches a UUID request-id to every request and logs method, URL, client IP,
headers, query params, and body on arrival — formatted as indented JSON.

For multipart/form-data bodies, only a non-consuming summary is logged
(parsing the form here would exhaust the stream before the route reads it);
raw bytes are never logged.

For JSON chat bodies, base64 media payloads (``input_audio.data`` and
``data:`` image URIs inside ``messages[].content[]``) are replaced with a
short size summary so logs aren't flooded with hundreds of KB of base64.
Response logging is handled by send_response (app/utils/response.py).
"""

from __future__ import annotations

import json
import uuid
from typing import Any

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


def _b64_size(data: str) -> str:
    """Summarize a (possibly ``data:...;base64,``) base64 string by decoded size."""
    if data.startswith("data:") and ";base64," in data:
        data = data.split(";base64,", 1)[1]
    approx_bytes = len(data) * 3 // 4
    if approx_bytes >= 1024:
        return f"~{approx_bytes // 1024} KB"
    return f"~{approx_bytes} B"


def _is_data_uri(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("data:") and ";base64," in value


def _sanitize_part(part: Any) -> Any:
    """Replace base64 media payloads in a single OpenAI content part with a summary."""
    if not isinstance(part, dict):
        return part

    if isinstance(part.get("input_audio"), dict) and part["input_audio"].get("data"):
        audio = {**part["input_audio"]}
        audio["data"] = f"<base64 audio, {_b64_size(str(audio['data']))}>"
        return {**part, "input_audio": audio}

    image_url = part.get("image_url")
    if isinstance(image_url, dict) and _is_data_uri(image_url.get("url")):
        return {**part, "image_url": {**image_url, "url": f"<base64 image, {_b64_size(image_url['url'])}>"}}
    if _is_data_uri(image_url):
        return {**part, "image_url": f"<base64 image, {_b64_size(image_url)}>"}
    if _is_data_uri(part.get("input_image")):
        return {**part, "input_image": f"<base64 image, {_b64_size(part['input_image'])}>"}

    return part


def _sanitize_messages(body: Any) -> Any:
    """Summarize base64 media inside ``messages[].content[]`` for logging.

    OpenAI-compatible payloads only carry base64 media in chat content parts, so
    we walk that fixed path rather than recursing the whole body. Copies are built
    instead of mutating, so the parsed body used by routes is never altered.
    """
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        return body

    messages = []
    for message in body["messages"]:
        if isinstance(message, dict) and isinstance(message.get("content"), list):
            content = [_sanitize_part(part) for part in message["content"]]
            messages.append({**message, "content": content})
        else:
            messages.append(message)
    return {**body, "messages": messages}


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
        # Parsing the form here (await request.form()) would consume the request
        # stream and leave the route handler with an empty body. Log a
        # non-consuming summary instead — raw bytes are never logged anyway.
        return f"<multipart/form-data, content-length={request.headers.get('content-length', 'unknown')}>"

    if "application/json" in content_type:
        try:
            raw = await request.body()
            return _sanitize_messages(json.loads(raw))
        except Exception:
            return "<unparseable JSON>"

    if "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        return dict(form)

    return None
