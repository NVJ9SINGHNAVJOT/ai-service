"""
Model management API routes.

GET    /api/v1/models                   → list all models
POST   /api/v1/models/load              → load a model into inference memory
POST   /api/v1/models/unload            → unload the current model
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException

from app.core.exceptions import (
    ModelLoadError,
    ModelNotFoundError,
)
from app.schemas.model import (
    APIResponse,
    LoadModelRequest,
    ModelInfo,
    UnloadModelRequest,
)
from app.services.model_manager import ModelManager

router = APIRouter(prefix="/api/v1/models", tags=["models"])
_manager = ModelManager()


def _load_model_into_memory(name: str) -> None:
    """
    Resolve a model by name and load it into the shared inference service.

    Keeping this in a small helper makes the route itself easier to read and
    avoids repeating the "look up path, convert to Path, then load" sequence.
    """
    from app.main import inference_service

    info = _manager.get_model(name)
    inference_service.load(
        model_path=Path(info.path),
        model_name=info.name,
    )


# ── List ─────────────────────────────────────────────────────────────────────

@router.get("", response_model=APIResponse, summary="List all local models")
async def list_models() -> APIResponse:
    """Return all models found in downloaded/ and custom/ directories."""
    try:
        models: List[ModelInfo] = _manager.list_models()
        return APIResponse(
            success=True,
            message=f"Found {len(models)} model(s).",
            data=[m.model_dump() for m in models],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Load ─────────────────────────────────────────────────────────────────────

@router.post("/load", response_model=APIResponse, summary="Load a model into memory")
async def load_model(body: LoadModelRequest) -> APIResponse:
    """
    Load the named model into memory for inference.

    If a different model is already loaded it will be swapped out automatically.
    """
    try:
        _load_model_into_memory(body.name)
        return APIResponse(
            success=True,
            message=f"Model '{body.name}' is now loaded.",
            data={"loaded_model": body.name},
        )
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ModelLoadError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Unload ───────────────────────────────────────────────────────────────────

@router.post("/unload", response_model=APIResponse, summary="Unload the current model from memory")
async def unload_model(body: UnloadModelRequest) -> APIResponse:
    """
    Unload the currently loaded model and free memory.

    If `name` is provided but does not match the currently loaded model,
    a 400 error is returned.
    """
    from app.main import inference_service

    current = inference_service.loaded_model_name

    if body.name and current and body.name != current:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{body.name}' is not currently loaded ('{current}' is).",
        )

    unloaded = inference_service.unload()
    if unloaded:
        return APIResponse(
            success=True,
            message=f"Model '{unloaded}' unloaded.",
            data={"unloaded_model": unloaded},
        )
    return APIResponse(
        success=True,
        message="No model was loaded.",
        data=None,
    )
