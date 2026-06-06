"""
Health check endpoint.

GET /health  →  {"status": "ok", "model_loaded": bool, "loaded_model": str | null}
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Minimal status payload used by uptime checks and automated tests."""

    status: str
    model_loaded: bool
    loaded_model: Optional[str] = None
    loaded_model_backend: Optional[str] = None


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health() -> HealthResponse:
    """Return server health and currently loaded model info."""
    from app.main import inference_service, media_inference_service  # imported here to avoid circular imports

    loaded_model = inference_service.loaded_model_name or media_inference_service.loaded_model_name
    model_loaded = inference_service.is_loaded or media_inference_service.is_loaded
    backend = (
        "mlx-lm" if inference_service.is_loaded else
        "mlx-vlm" if media_inference_service.is_loaded else
        None
    )

    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        loaded_model=loaded_model,
        loaded_model_backend=backend,
    )
