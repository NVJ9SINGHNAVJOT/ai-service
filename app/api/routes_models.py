"""
Model management API routes.

GET    /api/v1/models                   → list all models
POST   /api/v1/models/load              → load a model into inference memory
POST   /api/v1/models/unload            → unload the current model
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import JSONResponse

from app.api.response import send_response

from app.config import settings
from app.core.exceptions import (
    InvalidModelPathError,
    ModelLoadError,
    ModelNotFoundError,
    UnsupportedModelError,
)
from app.schemas.model import (
    APIResponse,
    LoadModelRequest,
    ModelInfo,
    ModelListResponse,
    UnloadModelRequest,
)
from app.services.model_manager import ModelManager

router = APIRouter(prefix="/api/v1/models", tags=["models"])
_manager = ModelManager()

# Reusable error envelope for OpenAPI response documentation.
_ERROR_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"detail": {"type": "string", "description": "Human-readable error message."}},
}

_LOAD_MODEL_EXAMPLES = {
    "text_model": {
        "summary": "Load a text model",
        "description": "Loads a text model into the shared inference memory, swapping out any other loaded model.",
        "value": {"name": settings.example_text_model},
    },
    "media_model": {
        "summary": "Load a multimodal model",
        "description": "Loads a vision/audio model used by image and audio chat completions.",
        "value": {"name": settings.example_media_model},
    },
}

_UNLOAD_MODEL_EXAMPLES = {
    "unload_current": {
        "summary": "Unload whatever is loaded",
        "description": "Omit `name` to unload whichever model is currently resident.",
        "value": {},
    },
    "unload_named": {
        "summary": "Unload a specific model (guardrail)",
        "description": "Pass `name` to assert which model you expect to unload; a mismatch returns HTTP 400.",
        "value": {"name": settings.example_text_model},
    },
    "unload_mismatch_400": {
        "summary": "Negative — name mismatch (400)",
        "description": "Requesting a name other than the currently loaded model returns HTTP 400.",
        "value": {"name": "some-other-model"},
    },
}


def _load_model_into_memory(name: str) -> None:
    """
    Resolve a model by name and load it into the correct inference service.

    Backend is selected from the model's `backend` field (derived from the
    conversion tool): mlx-vlm models go to media_inference_service, mlx-lm
    models go to inference_service.
    """
    from app.main import inference_service, media_inference_service

    raw_info = _manager.get_model(name)
    is_vlm = raw_info.backend == "mlx-vlm"

    if is_vlm:
        info = _manager.ensure_model_files_ready(name)
        inference_service.unload()
        media_inference_service.load(
            model_path=Path(info.path),
            model_name=info.name,
        )
    else:
        info = _manager.ensure_model_loadable(name)
        media_inference_service.unload()
        inference_service.load(
            model_path=Path(info.path),
            model_name=info.name,
        )


# ── List ─────────────────────────────────────────────────────────────────────

@router.get("", response_model=ModelListResponse, summary="List all local models")
async def list_models(request: Request) -> JSONResponse:
    """Return all models found in downloaded/ and custom/ directories."""
    try:
        models: List[ModelInfo] = _manager.list_models()
        return send_response(request, APIResponse(
            success=True,
            message=f"Found {len(models)} model(s).",
            data=[m.model_dump() for m in models],
        ))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Load ─────────────────────────────────────────────────────────────────────

@router.post(
    "/load",
    response_model=APIResponse,
    summary="Load a model into memory",
    responses={
        400: {
            "description": "The model files are incomplete or the architecture is unsupported.",
            "content": {"application/json": {"schema": _ERROR_RESPONSE_SCHEMA}},
        },
        404: {
            "description": "No local model with that name was found.",
            "content": {"application/json": {"schema": _ERROR_RESPONSE_SCHEMA}},
        },
        500: {
            "description": "The model failed to load into memory.",
            "content": {"application/json": {"schema": _ERROR_RESPONSE_SCHEMA}},
        },
    },
)
async def load_model(
    request: Request,
    body: LoadModelRequest = Body(..., openapi_examples=_LOAD_MODEL_EXAMPLES),
) -> JSONResponse:
    """
    Load the named model into memory for inference.

    If a different model is already loaded it will be swapped out automatically.
    """
    try:
        _load_model_into_memory(body.name)
        return send_response(request, APIResponse(
            success=True,
            message=f"Model '{body.name}' is now loaded.",
            data={"loaded_model": body.name},
        ))
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (InvalidModelPathError, UnsupportedModelError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ModelLoadError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Unload ───────────────────────────────────────────────────────────────────

@router.post(
    "/unload",
    response_model=APIResponse,
    summary="Unload the current model from memory",
    responses={
        400: {
            "description": "The provided `name` does not match the currently loaded model.",
            "content": {
                "application/json": {
                    "schema": _ERROR_RESPONSE_SCHEMA,
                    "example": {"detail": f"Model 'some-other-model' is not currently loaded ('{settings.example_text_model}' is)."},
                }
            },
        },
    },
)
async def unload_model(
    request: Request,
    body: UnloadModelRequest = Body(..., openapi_examples=_UNLOAD_MODEL_EXAMPLES),
) -> JSONResponse:
    """
    Unload the currently loaded model and free memory.

    If `name` is provided but does not match the currently loaded model,
    a 400 error is returned.
    """
    from app.main import inference_service, media_inference_service

    current = inference_service.loaded_model_name or media_inference_service.loaded_model_name

    if body.name and current and body.name != current:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{body.name}' is not currently loaded ('{current}' is).",
        )

    unloaded = inference_service.unload() or media_inference_service.unload()
    if unloaded:
        return send_response(request, APIResponse(
            success=True,
            message=f"Model '{unloaded}' unloaded.",
            data={"unloaded_model": unloaded},
        ))
    return send_response(request, APIResponse(
        success=True,
        message="No model was loaded.",
        data=None,
    ))
