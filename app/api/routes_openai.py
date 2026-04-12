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

from fastapi import APIRouter, HTTPException, Request
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


@router.post("/chat/completions", response_model=OpenAIChatCompletionResponse)
async def create_chat_completion(
    request: Request,
    body: OpenAIChatCompletionRequest,
):
    """
    OpenAI-compatible chat completions endpoint.

    Notes:
    - Usage fields are estimated until tokenizer-based accounting is added.
    """
    _reject_unsupported_chat_features(body)
    stop_sequences = _normalize_stop_sequences(body.stop)

    uses_vlm = _request_uses_vlm(body.messages)
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
        if body.verbose:
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
            text = "".join(chunks)
        else:
            if stop_sequences:
                chunks: list[str] = []
                usage = {}
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
                text = "".join(chunks)
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
