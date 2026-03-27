"""
Inference API routes.

POST /api/v1/inference/generate   → raw text generation
POST /api/v1/inference/chat       → chat-style completion
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.exceptions import InferenceError, ModelLoadError, ModelNotFoundError
from app.schemas.inference import (
    ChatRequest,
    GenerateData,
    GenerateRequest,
    GenerateResponse,
)
from app.services.model_manager import ModelManager

router = APIRouter(prefix="/api/v1/inference", tags=["inference"])
_manager = ModelManager()


def _ensure_model_loaded(model_name: str) -> None:
    """
    Auto-load the requested model if it is not already in memory.

    Raises HTTPException on not-found or load failure.
    """
    from app.main import inference_service

    if inference_service.loaded_model_name == model_name:
        return  # already loaded

    try:
        info = _manager.get_model(model_name)
        inference_service.load(
            model_path=Path(info.path),
            model_name=info.name,
        )
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ModelLoadError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _stream_payload(model: str, text: str, usage: dict | None) -> str:
    """
    Serialize one chunk for the custom streaming API.

    The legacy/custom API uses NDJSON instead of Server-Sent Events so simple
    tools like `curl -N` can consume it one JSON object per line.
    """
    payload = {
        "model": model,
        "text": text,
        "done": usage is not None,
        "usage": usage,
    }
    return json.dumps(payload) + "\n"


# ── Generate ─────────────────────────────────────────────────────────────────

@router.post("/generate", response_model=GenerateResponse, summary="Generate text completion")
async def generate(body: GenerateRequest):
    """
    Generate a text completion for the given prompt.

    The model is auto-loaded if not already in memory.
    Optionally prepend a system prompt to the user prompt.
    """
    _ensure_model_loaded(body.model)

    from app.main import inference_service

    # Build final prompt: prepend system prompt if provided
    prompt = body.prompt
    if body.system_prompt:
        prompt = f"System: {body.system_prompt}\n\nUser: {body.prompt}\nAssistant:"

    if body.stream:
        def event_stream():
            try:
                for chunk, usage in inference_service.generate_stream(
                    prompt=prompt,
                    max_tokens=body.max_tokens,
                    temperature=body.temperature,
                    top_p=body.top_p,
                    repetition_penalty=body.repetition_penalty,
                ):
                    yield _stream_payload(body.model, chunk, usage)
            except InferenceError as exc:
                error_payload = {"error": str(exc), "done": True}
                yield json.dumps(error_payload) + "\n"

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    try:
        text, usage = inference_service.generate(
            prompt=prompt,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            repetition_penalty=body.repetition_penalty,
        )
    except InferenceError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return GenerateResponse(
        success=True,
        message="Generation successful.",
        data=GenerateData(model=body.model, text=text).model_dump(),
    )


# ── Chat ─────────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=GenerateResponse, summary="Chat completion")
async def chat(body: ChatRequest):
    """
    Run a chat completion using a list of messages.

    Applies the model's chat template if available; falls back to a
    plain-text format otherwise.

    The model is auto-loaded if not already in memory.
    """
    _ensure_model_loaded(body.model)

    from app.main import inference_service

    if body.stream:
        def event_stream():
            try:
                for chunk, usage in inference_service.chat_stream(
                    messages=body.messages,
                    max_tokens=body.max_tokens,
                    temperature=body.temperature,
                    top_p=body.top_p,
                    repetition_penalty=body.repetition_penalty,
                ):
                    yield _stream_payload(body.model, chunk, usage)
            except InferenceError as exc:
                error_payload = {"error": str(exc), "done": True}
                yield json.dumps(error_payload) + "\n"

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    try:
        text, usage = inference_service.chat(
            messages=body.messages,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            repetition_penalty=body.repetition_penalty,
        )
    except InferenceError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return GenerateResponse(
        success=True,
        message="Chat completion successful.",
        data=GenerateData(model=body.model, text=text).model_dump(),
    )
