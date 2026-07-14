"""
OpenAI-compatible API routes.

Currently supported:
- POST /v1/chat/completions
"""

from __future__ import annotations

import base64
import binascii
import json
import time
import uuid
from pathlib import Path
from typing import Any, Generator

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.response import log_response, send_response

from app.config import settings
from app.core.exceptions import InferenceError, InvalidModelPathError, ModelLoadError, ModelNotFoundError, UnsupportedModelError
from app.core.logging import get_logger
from app.schemas.inference import (
    ChatMessage,
    OpenAIChatCompletionChoice,
    OpenAIChatCompletionMessage,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIResponseMetrics,
    OpenAIUsage,
)
from app.services.media_inference import _strip_audio_data_uri
from app.services.model_manager import ModelManager

router = APIRouter(prefix="/v1", tags=["openai-compatible"])
_manager = ModelManager()
logger = get_logger(__name__)
_IGNORED_CHAT_FIELDS = {
    "metadata",
    "store",
    "service_tier",
    "seed",
    "safety_identifier",
    "stream_options",
}

# Request scenarios surfaced in Swagger UI. Replace the placeholder model names with
# values that exist on your machine, and the image/audio placeholders with real
# base64-encoded media bytes, before sending.
_CHAT_COMPLETION_EXAMPLES = {
    "text": {
        "summary": "Text — basic completion",
        "description": "Basic text-only OpenAI-compatible chat completion.",
        "value": {
            "model": settings.example_text_model,
            "messages": [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "Say hello in one short sentence."},
            ],
        },
    },
    "developer_role_with_ignored_extras": {
        "summary": "Developer role + ignored OpenAI extras",
        "description": (
            "Exercises the OpenAI-compatible subset: the `developer` role, the "
            "`max_completion_tokens` alias, and safe extras (`store`, `metadata`, "
            "`service_tier`, `seed`, `safety_identifier`, `stream_options`) that "
            "are accepted but ignored locally."
        ),
        "value": {
            "model": settings.example_text_model,
            "messages": [
                {"role": "developer", "content": "You are terse and practical."},
                {"role": "user", "content": "Reply with one line."},
            ],
            "store": False,
            "metadata": {"origin": "swagger"},
            "service_tier": "default",
            "seed": 123,
            "safety_identifier": "local-user-1",
            "stream_options": {"include_usage": True},
            "n": 1,
            "max_completion_tokens": 128,
        },
    },
    "verbose": {
        "summary": "Verbose — return x_metrics",
        "description": "Returns server-side timing metrics in `x_metrics` alongside the normal completion output.",
        "value": {
            "model": settings.example_text_model,
            "messages": [{"role": "user", "content": "Write a short haiku about coding."}],
            "verbose": True,
        },
    },
    "stop_sequence": {
        "summary": "Stop sequence",
        "description": "OpenAI-style `stop` support. The response text is trimmed before the stop sequence.",
        "value": {
            "model": settings.example_text_model,
            "messages": [
                {"role": "user", "content": "Write a sentence that includes END and more text after it."}
            ],
            "stop": "END",
        },
    },
    "streaming": {
        "summary": "Streaming (SSE)",
        "description": (
            "Server-Sent Events streaming response. Send `Accept: text/event-stream` and inspect the "
            "raw streamed body; frames are `data: {chunk}` lines terminated by `data: [DONE]`."
        ),
        "value": {
            "model": settings.example_text_model,
            "messages": [{"role": "user", "content": "Stream a short reply."}],
            "stream": True,
            "verbose": True,
        },
    },
    "image": {
        "summary": "Image + text (multimodal)",
        "description": (
            "Image + text chat completion. Routed through mlx-vlm automatically because the "
            "message carries an image. `image_url.url` takes a **base64 data URI** of the form "
            "`data:<mime>;base64,<bytes>` — as shown below, produced by e.g. "
            "`'data:image/jpeg;base64,' + base64.b64encode(open('photo.jpg', 'rb').read()).decode()` "
            "— **or** an `http(s)://` URL. It is **not** a filesystem path. Replace the placeholder "
            "below with your own base64 string before sending."
        ),
        "value": {
            "model": settings.example_media_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail."},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<base64-encoded-image-bytes>"}},
                    ],
                }
            ],
        },
    },
    "audio": {
        "summary": "Audio + text (multimodal)",
        "description": (
            "Audio + text chat completion. Routed through mlx-vlm automatically because the "
            "message carries audio. `input_audio.data` must be **base64-encoded audio bytes** "
            "— exactly what the OpenAI SDK sends, e.g. "
            "`base64.b64encode(open('clip.wav', 'rb').read()).decode()` — not a file path. "
            "`format` is the source type such as `wav` or `mp3`. Replace the placeholder below "
            "with your own base64 string before sending."
        ),
        "value": {
            "model": settings.example_media_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Transcribe or summarize this clip."},
                        {"type": "input_audio", "input_audio": {"data": "<base64-encoded-wav-bytes>", "format": "wav"}},
                    ],
                }
            ],
        },
    },
    "image_streaming": {
        "summary": "Image + text, streaming",
        "description": (
            "Streaming multimodal image request (SSE). Same image contract as the non-streaming "
            "example: `image_url.url` takes a **base64 data URI** (`data:<mime>;base64,<bytes>`, "
            "as shown below) **or** an `http(s)://` URL — never a filesystem path. Replace the "
            "placeholder below with your own base64 string before sending."
        ),
        "value": {
            "model": settings.example_media_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What do you see here?"},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<base64-encoded-image-bytes>"}},
                    ],
                }
            ],
            "stream": True,
            "verbose": True,
        },
    },
    "tools_unsupported_400": {
        "summary": "Negative — tools unsupported (400)",
        "description": "`tools` is not supported by this local endpoint yet and returns HTTP 400.",
        "value": {
            "model": settings.example_text_model,
            "messages": [{"role": "user", "content": "Call a tool for this."}],
            "tools": [{"type": "function", "function": {"name": "lookup_weather"}}],
        },
    },
    "unknown_field_400": {
        "summary": "Negative — unknown extra field (400)",
        "description": "Unknown request fields are rejected with HTTP 400 instead of being silently ignored.",
        "value": {
            "model": settings.example_text_model,
            "messages": [{"role": "user", "content": "Hello"}],
            "totally_unknown_option": True,
        },
    },
    "n_greater_than_one_400": {
        "summary": "Negative — n > 1 unsupported (400)",
        "description": "Only a single completion choice is supported; `n` > 1 returns HTTP 400.",
        "value": {
            "model": settings.example_text_model,
            "messages": [{"role": "user", "content": "Hello"}],
            "n": 2,
        },
    },
}

# Reusable error envelope for OpenAPI response documentation.
_ERROR_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"detail": {"type": "string", "description": "Human-readable error message."}},
}
# Response examples for the 200 case. Injected into the OpenAPI schema via a
# custom app.openapi() in app/main.py rather than the `responses=` parameter,
# because FastAPI serialises the whole schema with exclude_none=True and would
# otherwise strip the `x_metrics: null` key from the non-verbose example.
CHAT_COMPLETION_200_EXAMPLES = {
    "text": {
        "summary": "Without verbose — x_metrics is null",
        "value": {
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": settings.example_text_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello! How can I help you today?"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 9, "total_tokens": 19},
            "x_metrics": None,
        },
    },
    "verbose": {
        "summary": "With verbose=true — x_metrics populated",
        "value": {
            "id": "chatcmpl-abc456",
            "object": "chat.completion",
            "created": 1700000000,
            "model": settings.example_text_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Bits cascade down,\nFunctions bloom like cherry trees,\nBug fixed at midnight."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 18, "total_tokens": 26},
            "x_metrics": {
                "total_duration_s": 3.21,
                "load_duration_s": 0.45,
                "prompt_eval_count": 8,
                "prompt_eval_duration_s": 0.12,
                "prompt_eval_rate": 66.7,
                "eval_count": 18,
                "eval_duration_s": 2.64,
                "eval_rate": 6.8,
            },
        },
    },
}

_CHAT_COMPLETION_RESPONSES = {
    400: {
        "description": (
            "Validation error — an unsupported or malformed field was sent "
            "(e.g. `tools`, `n` > 1, an unknown field, an invalid `stop`, or a media input "
            "that is not a valid OpenAI form — such as a filesystem path)."
        ),
        "content": {
            "application/json": {
                "schema": _ERROR_RESPONSE_SCHEMA,
                "example": {"detail": "Field 'tools' is not supported by this local chat completions endpoint yet."},
            }
        },
    },
    404: {
        "description": "The requested model was not found locally.",
        "content": {
            "application/json": {
                "schema": _ERROR_RESPONSE_SCHEMA,
                "example": {"detail": f"Model not found: '{settings.example_text_model}'"},
            }
        },
    },
    500: {
        "description": "The model failed to load or inference failed.",
        "content": {
            "application/json": {
                "schema": _ERROR_RESPONSE_SCHEMA,
                "example": {"detail": "Inference error: <backend message>"},
            }
        },
    },
}


def _ensure_model_loaded(model_name: str) -> None:
    """Auto-load the requested model if needed."""
    from app.main import inference_service, media_inference_service

    if inference_service.loaded_model_name == model_name:
        return

    try:
        info = _manager.ensure_model_loadable(model_name)
        media_inference_service.unload()
        inference_service.load(
            model_path=Path(info.path),
            model_name=info.name,
        )
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (InvalidModelPathError, UnsupportedModelError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ModelLoadError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _ensure_media_model_loaded(model_name: str) -> None:
    """Auto-load the requested model into the shared mlx-vlm service if needed."""
    from app.main import inference_service, media_inference_service

    if media_inference_service.loaded_model_name == model_name:
        return

    try:
        info = _manager.ensure_model_files_ready(model_name)
        inference_service.unload()
        media_inference_service.load(
            model_path=Path(info.path),
            model_name=info.name,
        )
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InvalidModelPathError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ModelLoadError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _estimate_usage(prompt_text: str, completion_text: str) -> OpenAIUsage:
    """
    Provide a lightweight usage estimate.

    MLX-LM does not currently return token counts here, so we use a simple
    whitespace-based approximation to keep the OpenAI response shape intact.
    """
    prompt_tokens = len(prompt_text.split())
    completion_tokens = len(completion_text.split())
    return OpenAIUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _reject_unsupported_chat_features(body: OpenAIChatCompletionRequest) -> None:
    """Validate which OpenAI-compatible fields we actually support today."""
    extras = getattr(body, "__pydantic_extra__", {}) or {}
    unknown_fields = sorted(name for name in extras if name not in _IGNORED_CHAT_FIELDS)
    if unknown_fields:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported OpenAI chat completions field(s): "
                + ", ".join(unknown_fields)
            ),
        )

    if body.n not in (None, 1):
        raise HTTPException(
            status_code=400,
            detail="Field 'n' is supported only with the value 1.",
        )

    if body.max_tokens is not None and body.max_completion_tokens is not None and body.max_tokens != body.max_completion_tokens:
        raise HTTPException(
            status_code=400,
            detail="Fields 'max_tokens' and 'max_completion_tokens' must match when both are provided.",
        )

    if body.tools is not None:
        raise HTTPException(status_code=400, detail="Field 'tools' is not supported by this local chat completions endpoint yet.")
    if body.tool_choice is not None:
        raise HTTPException(status_code=400, detail="Field 'tool_choice' is not supported by this local chat completions endpoint yet.")
    if body.parallel_tool_calls is not None:
        raise HTTPException(status_code=400, detail="Field 'parallel_tool_calls' is not supported by this local chat completions endpoint yet.")
    if body.function_call is not None:
        raise HTTPException(status_code=400, detail="Field 'function_call' is not supported by this local chat completions endpoint yet.")
    if body.stop is not None:
        stop_sequences = _normalize_stop_sequences(body.stop)
        if not stop_sequences:
            raise HTTPException(status_code=400, detail="Field 'stop' must be a non-empty string or a list of non-empty strings.")
        if len(stop_sequences) > 4:
            raise HTTPException(status_code=400, detail="Field 'stop' supports at most 4 stop sequences.")
    if body.prediction is not None:
        raise HTTPException(status_code=400, detail="Field 'prediction' is not supported by this local chat completions endpoint yet.")
    if body.audio is not None:
        raise HTTPException(status_code=400, detail="Field 'audio' is not supported because this endpoint does not generate audio output.")
    if body.logprobs:
        raise HTTPException(status_code=400, detail="Field 'logprobs' is not supported by this local chat completions endpoint yet.")
    if body.top_logprobs is not None:
        raise HTTPException(status_code=400, detail="Field 'top_logprobs' is not supported by this local chat completions endpoint yet.")
    if body.frequency_penalty not in (None, 0, 0.0):
        raise HTTPException(status_code=400, detail="Field 'frequency_penalty' is not supported by this local chat completions endpoint yet.")
    if body.presence_penalty not in (None, 0, 0.0):
        raise HTTPException(status_code=400, detail="Field 'presence_penalty' is not supported by this local chat completions endpoint yet.")

    if body.response_format is not None:
        format_type = body.response_format.get("type") if isinstance(body.response_format, dict) else None
        if format_type not in (None, "text"):
            raise HTTPException(
                status_code=400,
                detail="Field 'response_format' is only supported with type='text' on this local chat completions endpoint.",
            )

    if body.modalities is not None:
        unsupported_modalities = [modality for modality in body.modalities if modality != "text"]
        if unsupported_modalities:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Field 'modalities' is only supported with ['text'] on this local chat completions endpoint."
                ),
            )


def _decode_inline_base64(payload: str, detail: str) -> None:
    """Validate that `payload` is non-empty base64, raising 400 with `detail` if not.

    `detail` describes the expected shape only — it never carries the payload, so a
    multi-megabyte body can't be echoed into the response or the logs.
    """
    try:
        raw = base64.b64decode("".join(payload.split()), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=detail) from exc
    if not raw:
        raise HTTPException(status_code=400, detail=detail)


def _iter_media_parts(message: ChatMessage) -> Generator[tuple[str, str, Any], None, None]:
    """Yield ``(kind, field, value)`` for every media content part, as sent.

    Deliberately reads the raw content parts rather than ``image_inputs()`` /
    ``audio_inputs()``: those accessors drop parts whose value is empty, so an
    empty url or data would never reach validation — it would be silently ignored
    and the request would answer from the remaining text alone.
    """
    if not isinstance(message.content, list):
        return

    for item in message.content:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        if item_type == "image_url":
            payload = item.get("image_url")
            yield "image", "image_url.url", payload.get("url") if isinstance(payload, dict) else payload
        elif item_type == "input_image":
            yield "image", "input_image.image_url", item.get("image_url")
        elif item_type == "input_audio":
            payload = item.get("input_audio")
            yield "audio", "input_audio.data", payload.get("data") if isinstance(payload, dict) else None


def _reject_unsupported_media_inputs(body: OpenAIChatCompletionRequest) -> None:
    """Reject media forms that are not valid OpenAI — notably filesystem paths.

    Images may be a base64 data URI or an `http(s)://` URL (both are valid OpenAI,
    and mlx-vlm handles either); audio is base64 only, since the Chat Completions
    `input_audio` part has no URL form. Failing here keeps a bad payload from
    reaching mlx-vlm, whose errors embed the entire source string.

    The CLI is unaffected: it talks to the services directly and keeps its file paths.
    """
    for message in body.messages:
        for kind, field, value in _iter_media_parts(message):
            if not isinstance(value, str) or not value.strip():
                raise HTTPException(
                    status_code=400,
                    detail=f"{field} must be a non-empty string.",
                )

            if kind == "audio":
                _decode_inline_base64(
                    _strip_audio_data_uri(value),
                    f"{field} must be base64-encoded audio bytes (as the OpenAI SDK sends).",
                )
                continue

            if value.startswith(("http://", "https://")):
                continue
            if not value.startswith("data:image/") or ";base64," not in value:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"{field} must be a base64 data URI of the form "
                        "'data:image/<subtype>;base64,<bytes>' or an http(s):// URL. "
                        "Filesystem paths are not supported."
                    ),
                )
            _decode_inline_base64(
                value.split(";base64,", 1)[1],
                f"{field} must carry valid, non-empty base64-encoded image bytes.",
            )


def _request_uses_vlm(messages) -> bool:
    """Return True when any chat message includes image or audio content."""
    return any(message.has_image() or message.has_audio() for message in messages)


def _model_is_vlm(model_name: str) -> bool:
    """Return True when the model was converted with mlx-vlm (backend field)."""
    try:
        info = _manager.get_model(model_name)
        return info.backend == "mlx-vlm"
    except ModelNotFoundError:
        # Unknown model — the loader raises the 404 (logged at the boundary)
        # later; default to the text backend here without extra noise.
        return False
    except Exception as exc:
        # A real lookup failure (e.g. registry read error) must not silently
        # misroute to the text backend without leaving a trace.
        logger.warning(
            "Backend detection failed for '%s' (%s: %s); defaulting to text backend.",
            model_name, type(exc).__name__, exc,
        )
        return False


def _messages_to_prompt_text(messages) -> str:
    """Flatten only the textual portions of chat messages for fallback usage estimates."""
    return "\n".join(message.text_content() for message in messages)


def _normalize_stop_sequences(stop: str | list[str] | None) -> list[str]:
    """Normalize the OpenAI `stop` field into a clean list of non-empty strings."""
    if stop is None:
        return []
    if isinstance(stop, str):
        raw_items = [stop]
    else:
        raw_items = stop
    return [item for item in raw_items if isinstance(item, str) and item]


def _split_at_stop_sequence(text: str, stop_sequences: list[str]) -> tuple[str, bool]:
    """Return text trimmed at the earliest stop sequence, if any."""
    if not stop_sequences:
        return text, False

    first_index: int | None = None
    for stop_sequence in stop_sequences:
        index = text.find(stop_sequence)
        if index >= 0 and (first_index is None or index < first_index):
            first_index = index

    if first_index is None:
        return text, False
    return text[:first_index], True


def _stream_with_stop_sequences(chunks_with_usage, stop_sequences: list[str]):
    """
    Apply stop sequence trimming to a streaming iterator.

    This keeps enough trailing text buffered to detect stop sequences that span
    chunk boundaries.
    """
    if not stop_sequences:
        yield from chunks_with_usage
        return

    max_stop_len = max(len(item) for item in stop_sequences)
    pending = ""

    for chunk, usage in chunks_with_usage:
        pending += chunk
        trimmed, found_stop = _split_at_stop_sequence(pending, stop_sequences)
        if found_stop:
            final_usage = {"finish_reason": "stop"} if usage is None else {**usage, "finish_reason": "stop"}
            yield trimmed, final_usage
            return

        safe_len = max(0, len(pending) - (max_stop_len - 1))
        if safe_len > 0:
            safe_text = pending[:safe_len]
            pending = pending[safe_len:]
            yield safe_text, usage

    if pending:
        yield pending, None


def _collect_chat_completion(active_service, body, stop_sequences: list[str]) -> tuple[str, dict]:
    """
    Buffer a chat completion by draining the streaming path.

    Used whenever we need per-token handling for a non-streaming response —
    i.e. when `verbose` metrics or `stop` sequences are requested — so stop
    trimming and usage accounting stay identical to the streaming path.
    """
    chunks: list[str] = []
    usage: dict = {}
    stream_iter = active_service.chat_stream(
        messages=body.messages,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        repetition_penalty=body.repetition_penalty,
    )
    for chunk, stream_usage in _stream_with_stop_sequences(stream_iter, stop_sequences):
        chunks.append(chunk)
        if stream_usage is not None:
            usage = stream_usage
    return "".join(chunks), usage


def _sse_chunk(payload: dict) -> str:
    """Format one Server-Sent Events frame for OpenAI-style streaming."""
    return f"data: {json.dumps(payload)}\n\n"


def _build_verbose_metrics(usage: dict | None, load_duration_s: float | None) -> OpenAIResponseMetrics:
    """Convert internal usage metadata into a stable API metrics shape."""
    usage = usage or {}
    metrics = usage.get("metrics") or {}
    return OpenAIResponseMetrics(
        total_duration_s=metrics.get("total_duration_s"),
        load_duration_s=load_duration_s,
        prompt_eval_count=usage.get("prompt_tokens"),
        prompt_eval_duration_s=metrics.get("prompt_eval_duration_s"),
        prompt_eval_rate=metrics.get("prompt_eval_rate"),
        eval_count=usage.get("completion_tokens"),
        eval_duration_s=metrics.get("eval_duration_s"),
        eval_rate=metrics.get("eval_rate"),
    )


@router.post(
    "/chat/completions",
    response_model=OpenAIChatCompletionResponse,
    summary="Create a chat completion",
    responses=_CHAT_COMPLETION_RESPONSES,
)
async def create_chat_completion(
    request: Request,
    body: OpenAIChatCompletionRequest = Body(..., openapi_examples=_CHAT_COMPLETION_EXAMPLES),
):
    """
    OpenAI-compatible chat completions endpoint.

    Supports text, multimodal (image / audio), streaming (SSE), `verbose`
    timing metrics, and OpenAI-style `stop` sequences. Image- or audio-bearing
    requests are routed through mlx-vlm automatically; everything else uses
    mlx-lm. The requested `model` is auto-loaded (swapping out any other loaded
    model) before generation.

    Media follows the OpenAI contract: `image_url.url` is a base64 data URI
    (`data:image/<subtype>;base64,<bytes>`) or an `http(s)://` URL, and
    `input_audio.data` is base64 bytes (audio has no URL form). Filesystem paths
    return 400.

    Use the **Examples** dropdown above to try each scenario, including the
    negative cases that return HTTP 400.

    Notes:
    - Usage fields are estimated until tokenizer-based accounting is added.
    """
    _reject_unsupported_chat_features(body)
    _reject_unsupported_media_inputs(body)
    stop_sequences = _normalize_stop_sequences(body.stop)

    uses_vlm = _request_uses_vlm(body.messages) or _model_is_vlm(body.model)
    if uses_vlm:
        from app.main import media_inference_service as active_service
        already_loaded = active_service.loaded_model_name == body.model
        _ensure_media_model_loaded(body.model)
    else:
        from app.main import inference_service as active_service
        already_loaded = active_service.loaded_model_name == body.model
        _ensure_model_loaded(body.model)

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    # load_duration is per-turn: the real load cost when this request loaded the model,
    # omitted (None) on a warm turn where the model was already resident.
    load_duration_s = None if already_loaded else active_service.last_load_duration_s
    # Token counts (usage) are emitted whenever the client opts in via stream_options, regardless
    # of verbose; verbose only adds the richer x_metrics timing block.
    include_usage = bool((body.stream_options or {}).get("include_usage"))

    if body.stream:
        async def sse_stream():
            # Accumulate each chunk's parsed payload (plus the terminal [DONE]
            # marker) so the full stream can be logged as pretty JSON once it
            # ends. The `finally` also fires on early returns / disconnects,
            # capturing partial streams.
            frames: list[dict | str] = []

            def emit_chunk(payload: dict) -> str:
                frames.append(payload)
                return _sse_chunk(payload)

            def emit_done() -> str:
                frames.append("[DONE]")
                return "data: [DONE]\n\n"

            try:
                initial_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": body.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
                yield emit_chunk(initial_chunk)
                if await request.is_disconnected():
                    return

                stream_iter = active_service.chat_stream(
                    messages=body.messages,
                    max_tokens=body.max_tokens,
                    temperature=body.temperature,
                    top_p=body.top_p,
                    repetition_penalty=body.repetition_penalty,
                )
                final_usage: dict | None = None
                for chunk, usage in _stream_with_stop_sequences(stream_iter, stop_sequences):
                    if await request.is_disconnected():
                        return
                    if usage is not None:
                        final_usage = usage
                    payload = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": body.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk} if chunk else {},
                                "finish_reason": None if usage is None else usage.get("finish_reason", "stop"),
                            }
                        ],
                    }
                    if body.verbose and usage is not None:
                        payload["x_metrics"] = _build_verbose_metrics(usage, load_duration_s).model_dump()
                    yield emit_chunk(payload)
                    if await request.is_disconnected():
                        return

                # OpenAI-style final usage chunk (empty choices), sent whenever the client opted in
                # via stream_options.include_usage — independent of verbose.
                if include_usage and final_usage is not None:
                    usage_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": body.model,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": int(final_usage.get("prompt_tokens", 0)),
                            "completion_tokens": int(final_usage.get("completion_tokens", 0)),
                            "total_tokens": int(final_usage.get("total_tokens", 0)),
                        },
                    }
                    yield emit_chunk(usage_chunk)

                yield emit_done()
            except Exception as exc:
                # The stream has already started, so the central exception
                # handlers in app/main.py can no longer catch this — the SSE
                # generator is its own boundary. Log once here (with traceback)
                # so a mid-stream failure is never silent, then surface a clean
                # error frame to the client.
                # request_id/correlation_id are added by the log formatter's
                # context prefix (the stream runs within the request's context).
                logger.error(
                    "Streaming chat completion failed | %s: %s",
                    type(exc).__name__, exc, exc_info=exc,
                )
                error_payload = {"error": {"message": str(exc), "type": "server_error"}}
                yield emit_chunk(error_payload)
                yield emit_done()
            finally:
                log_response(request, frames)

        return StreamingResponse(sse_stream(), media_type="text/event-stream")

    try:
        # `verbose` and `stop` both need per-token handling, so we drain the
        # streaming path; otherwise the buffered chat() call is the fast path.
        if body.verbose or stop_sequences:
            text, usage = _collect_chat_completion(active_service, body, stop_sequences)
        else:
            text, usage = active_service.chat(
                messages=body.messages,
                max_tokens=body.max_tokens,
                temperature=body.temperature,
                top_p=body.top_p,
                repetition_penalty=body.repetition_penalty,
            )
    except InferenceError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if usage:
        response_usage = OpenAIUsage(
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
        )
    else:
        prompt_text = _messages_to_prompt_text(body.messages)
        response_usage = _estimate_usage(prompt_text=prompt_text, completion_text=text)

    return send_response(request, OpenAIChatCompletionResponse(
        id=completion_id,
        object="chat.completion",
        created=created,
        model=body.model,
        choices=[
            OpenAIChatCompletionChoice(
                index=0,
                message=OpenAIChatCompletionMessage(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage=response_usage,
        x_metrics=_build_verbose_metrics(usage, load_duration_s) if body.verbose else None,
    ))
