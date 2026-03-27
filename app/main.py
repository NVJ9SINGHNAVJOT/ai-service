"""
FastAPI application entry point.

The `inference_service` module-level singleton is shared by the route modules
via a lazy import (to avoid circular imports at load time).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.core.exceptions import MLXManagerError
from app.core.logging import get_logger, setup_logging
from app.services.inference_service import InferenceService

setup_logging()
logger = get_logger(__name__)

# Module-level singleton shared by all route handlers
inference_service = InferenceService(cfg=settings)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan handler.

    On startup: ensure model directories exist and optionally pre-load
    the default model specified in settings.
    On shutdown: unload any loaded model.
    """
    settings.ensure_directories()
    logger.info("AI Service starting up.")

    if settings.default_model:
        from pathlib import Path
        from app.services.model_manager import ModelManager
        from app.core.exceptions import ModelNotFoundError, ModelLoadError

        try:
            manager = ModelManager(cfg=settings)
            info = manager.get_model(settings.default_model)
            inference_service.load(Path(info.path), info.name)
            logger.info("Default model '%s' pre-loaded.", settings.default_model)
        except (ModelNotFoundError, ModelLoadError) as exc:
            logger.warning("Could not pre-load default model: %s", exc)

    yield  # application runs here

    # Shutdown
    name = inference_service.unload()
    if name:
        logger.info("Model '%s' unloaded on shutdown.", name)
    logger.info("AI Service shut down.")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="AI Service",
        description="Local LLM management and inference server for Apple Silicon.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS — allow all origins for local development; tighten in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Domain exception → HTTP error
    @app.exception_handler(MLXManagerError)
    async def mlx_exception_handler(request, exc: MLXManagerError) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(exc), "data": None},
        )

    # Register routers
    from app.api.routes_health import router as health_router
    from app.api.routes_models import router as models_router
    from app.api.routes_inference import router as inference_router
    from app.api.routes_openai import router as openai_router

    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(inference_router)
    app.include_router(openai_router)

    return app


# The ASGI app instance used by uvicorn
app = create_app()
