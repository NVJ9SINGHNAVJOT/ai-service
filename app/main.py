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
from app.services.media_inference_service import MediaInferenceService

setup_logging()
logger = get_logger(__name__)

# Module-level singleton shared by all route handlers
inference_service = InferenceService(cfg=settings)
media_inference_service = MediaInferenceService(cfg=settings)


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
            is_vlm = info.backend == "mlx-vlm"
            if is_vlm:
                media_inference_service.load(Path(info.path), info.name)
            else:
                inference_service.load(Path(info.path), info.name)
            logger.info("Default model '%s' pre-loaded.", settings.default_model)
        except (ModelNotFoundError, ModelLoadError) as exc:
            logger.warning("Could not pre-load default model: %s", exc)

    yield  # application runs here

    # Shutdown
    name = inference_service.unload()
    if name:
        logger.info("Model '%s' unloaded on shutdown.", name)
    media_name = media_inference_service.unload()
    if media_name:
        logger.info("Media model '%s' unloaded on shutdown.", media_name)
    logger.info("AI Service shut down.")


_API_DESCRIPTION = """
Local LLM management and inference server for Apple Silicon (MLX).

**OpenAI-compatible** chat completions plus model lifecycle management. Every
endpoint below ships interactive **Examples** that mirror the Postman collection
(`postman/ai-service.postman_collection.json`), including the negative cases that
return HTTP 400 — pick one from the *Examples* dropdown and hit **Try it out**.

### Highlights
- `POST /v1/chat/completions` — text, image, and audio chat; SSE streaming;
  `verbose` timing metrics; OpenAI-style `stop` sequences.
- `POST /api/v1/models/load` · `/unload` — swap the in-memory model.
- `GET /api/v1/models` — list local models with state and input modalities.

> Replace placeholder model names, image paths, and audio data in the examples
> with values that exist on your machine before sending a request.
"""

_OPENAPI_TAGS = [
    {"name": "health", "description": "Liveness check and currently loaded model."},
    {"name": "models", "description": "List, load, and unload local models."},
    {
        "name": "openai-compatible",
        "description": "OpenAI-compatible chat completions (text, multimodal, streaming).",
    },
]


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="AI Service",
        description=_API_DESCRIPTION,
        version="1.0.0",
        lifespan=lifespan,
        openapi_tags=_OPENAPI_TAGS,
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
    from app.api.routes_openai import router as openai_router

    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(openai_router)

    return app


# The ASGI app instance used by uvicorn
app = create_app()
