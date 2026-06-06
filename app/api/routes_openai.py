"""
OpenAI-compatible API routes.

Currently supported:
- POST /v1/chat/completions
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.core.exceptions import InferenceError, InvalidModelPathError, ModelLoadError, ModelNotFoundError, UnsupportedModelError
from app.schemas.inference import (
    OpenAIChatCompletionChoice,
    OpenAIChatCompletionMessage,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIResponseMetrics,
    OpenAIUsage,
)
from app.services.model_manager import ModelManager

router = APIRouter(prefix="/v1", tags=["openai-compatible"])
_manager = ModelManager()
_IGNORED_CHAT_FIELDS = {
    "metadata",
    "store",
    "service_tier",
    "seed",
    "safety_identifier",
    "stream_options",
}

# Request scenarios surfaced in Swagger UI. These mirror the Postman collection
# (postman/ai-service.postman_collection.json) so the interactive docs and the
# Postman runner stay in lock-step. Replace the placeholder model names and image
# path with values that exist on your machine, and the audio data with real
# base64-encoded audio bytes, before sending.
_CHAT_COMPLETION_EXAMPLES = {
    "text": {
        "summary": "Text — basic completion",
        "description": "Basic text-only OpenAI-compatible chat completion.",
        "value": {
            "model": "mlx-community__Llama-3.2-3B-Instruct-4bit",
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
            "model": "mlx-community__Llama-3.2-3B-Instruct-4bit",
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
            "model": "mlx-community__Llama-3.2-3B-Instruct-4bit",
            "messages": [{"role": "user", "content": "Write a short haiku about coding."}],
            "verbose": True,
        },
    },
    "stop_sequence": {
        "summary": "Stop sequence",
        "description": "OpenAI-style `stop` support. The response text is trimmed before the stop sequence.",
        "value": {
            "model": "mlx-community__Llama-3.2-3B-Instruct-4bit",
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
            "model": "mlx-community__Llama-3.2-3B-Instruct-4bit",
            "messages": [{"role": "user", "content": "Stream a short reply."}],
            "stream": True,
            "verbose": True,
        },
    },
    "image": {
        "summary": "Image + text (multimodal)",
        "description": "Image + text chat completion. Routed through mlx-vlm automatically because the message carries an image.",
        "value": {
            "model": "mlx-community__gemma-4-e4b-bf16",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail."},
                        {"type": "image_url", "image_url": {"url": "/absolute/path/to/image.jpg"}},
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
            "model": "mlx-community__gemma-4-e4b-bf16",
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
        "description": "Streaming multimodal image request (SSE).",
        "value": {
            "model": "mlx-community__gemma-4-e4b-bf16",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What do you see here?"},
                        {"type": "image_url", "image_url": {"url": "/absolute/path/to/image.jpg"}},
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
            "model": "mlx-community__Llama-3.2-3B-Instruct-4bit",
            "messages": [{"role": "user", "content": "Call a tool for this."}],
            "tools": [{"type": "function", "function": {"name": "lookup_weather"}}],
        },
    },
    "unknown_field_400": {
        "summary": "Negative — unknown extra field (400)",
        "description": "Unknown request fields are rejected with HTTP 400 instead of being silently ignored.",
        "value": {
            "model": "mlx-community__Llama-3.2-3B-Instruct-4bit",
            "messages": [{"role": "user", "content": "Hello"}],
            "totally_unknown_option": True,
        },
    },
    "n_greater_than_one_400": {
        "summary": "Negative — n > 1 unsupported (400)",
        "description": "Only a single completion choice is supported; `n` > 1 returns HTTP 400.",
        "value": {
            "model": "mlx-community__Llama-3.2-3B-Instruct-4bit",
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
_CHAT_COMPLETION_RESPONSES = {
    400: {
        "description": (
            "Validation error — an unsupported or malformed field was sent "
            "(e.g. `tools`, `n` > 1, an unknown field, or an invalid `stop`)."
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
                "example": {"detail": "Model not found: 'mlx-community__Llama-3.2-3B-Instruct-4bit'"},
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


def _request_uses_vlm(messages) -> bool:
    """Return True when any chat message includes image or audio content."""
    return any(message.has_image() or message.has_audio() for message in messages)


def _model_is_vlm(model_name: str) -> bool:
    """Return True when the model was converted with mlx-vlm (backend field)."""
    try:
        info = _manager.get_model(model_name)
        return info.backend == "mlx-vlm"
    except Exception:
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

    Use the **Examples** dropdown above to try each scenario — they mirror the
    Postman collection, including the negative cases that return HTTP 400.

    Notes:
    - Usage fields are estimated until tokenizer-based accounting is added.
    """
    _reject_unsupported_chat_features(body)
    stop_sequences = _normalize_stop_sequences(body.stop)

    uses_vlm = _request_uses_vlm(body.messages) or _model_is_vlm(body.model)
    if uses_vlm:
        _ensure_media_model_loaded(body.model)
        from app.main import media_inference_service as active_service
    else:
        _ensure_model_loaded(body.model)
        from app.main import inference_service as active_service

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    load_duration_s = active_service.last_load_duration_s

    if body.stream:
        async def sse_stream():
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
                yield _sse_chunk(initial_chunk)
                if await request.is_disconnected():
                    return

                stream_iter = active_service.chat_stream(
                    messages=body.messages,
                    max_tokens=body.max_tokens,
                    temperature=body.temperature,
                    top_p=body.top_p,
                    repetition_penalty=body.repetition_penalty,
                )
                for chunk, usage in _stream_with_stop_sequences(stream_iter, stop_sequences):
                    if await request.is_disconnected():
                        return
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
                    yield _sse_chunk(payload)
                    if await request.is_disconnected():
                        return

                yield "data: [DONE]\n\n"
            except InferenceError as exc:
                error_payload = {"error": {"message": str(exc), "type": "server_error"}}
                yield _sse_chunk(error_payload)
                yield "data: [DONE]\n\n"

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

    return OpenAIChatCompletionResponse(
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
    )
