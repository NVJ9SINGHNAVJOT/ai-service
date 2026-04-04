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

from fastapi import APIRouter, HTTPException
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


def _ensure_model_loaded(model_name: str) -> None:
    """Auto-load the requested model if needed."""
    from app.main import inference_service

    if inference_service.loaded_model_name == model_name:
        return

    try:
        info = _manager.ensure_model_loadable(model_name)
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
    body: OpenAIChatCompletionRequest,
):
    """
    OpenAI-compatible chat completions endpoint.

    Notes:
    - Usage fields are estimated until tokenizer-based accounting is added.
    """
    _ensure_model_loaded(body.model)

    from app.main import inference_service

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    load_duration_s = inference_service.last_load_duration_s

    if body.stream:
        def sse_stream():
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

                for chunk, usage in inference_service.chat_stream(
                    messages=body.messages,
                    max_tokens=body.max_tokens,
                    temperature=body.temperature,
                    top_p=body.top_p,
                    repetition_penalty=body.repetition_penalty,
                ):
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
            for chunk, stream_usage in inference_service.chat_stream(
                messages=body.messages,
                max_tokens=body.max_tokens,
                temperature=body.temperature,
                top_p=body.top_p,
                repetition_penalty=body.repetition_penalty,
            ):
                chunks.append(chunk)
                if stream_usage is not None:
                    usage = stream_usage
            text = "".join(chunks)
        else:
            text, usage = inference_service.chat(
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
        prompt_text = "\n".join(message.content for message in body.messages)
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
