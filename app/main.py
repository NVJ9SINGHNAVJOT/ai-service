"""
FastAPI application entry point.

The `inference_service` module-level singleton is shared by the route modules
via a lazy import (to avoid circular imports at load time).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import settings
from app.core.exceptions import MLXManagerError
from app.core.logging import get_logger, setup_logging
from app.services.audio import AudioService
from app.services.inference import InferenceService
from app.services.media_inference import MediaInferenceService
from app.api.response import get_request_id, log_response, send_response

setup_logging()
logger = get_logger(__name__)

# Module-level singleton shared by all route handlers
inference_service = InferenceService(cfg=settings)
media_inference_service = MediaInferenceService(cfg=settings)
audio_service = AudioService(cfg=settings)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan handler.

    On startup: ensure model directories exist.
    On shutdown: unload any loaded model.
    """
    settings.ensure_directories()
    logger.info("AI Service starting up.")

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
endpoint below ships interactive **Examples**, including the negative cases that
return HTTP 400 — pick one from the *Examples* dropdown and hit **Try it out**.

### Highlights
- `POST /v1/chat/completions` — text, image, and audio chat; SSE streaming;
  `verbose` timing metrics; OpenAI-style `stop` sequences.
- `POST /v1/audio/transcriptions` · `/v1/audio/speech` — local speech-to-text (Whisper) and text-to-speech (Kokoro), fully on-device.
- `POST /api/v1/models/load` · `/unload` — swap the in-memory model.
- `GET /api/v1/models` — list local models with state, `backend` (mlx-lm / mlx-vlm), and `input_modalities` (text / image / audio / video).

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
    {
        "name": "audio",
        "description": "Local OpenAI-compatible speech-to-text (Whisper) and text-to-speech (Kokoro).",
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

    from app.api.middleware import LoggingMiddleware
    app.add_middleware(LoggingMiddleware)

    # Errors are logged once, here at the boundary, correlated by request_id.
    # request.state.request_id is set by LoggingMiddleware before routing.

    # Domain exception → HTTP 500, logged with traceback.
    @app.exception_handler(MLXManagerError)
    async def mlx_exception_handler(request: Request, exc: MLXManagerError) -> JSONResponse:
        logger.error(
            "Request failed | request_id=%s | %s: %s",
            get_request_id(request), type(exc).__name__, exc, exc_info=exc,
        )
        # Log the outgoing error response too, so the request flow stays
        # symmetric (Request received → Request failed → Response sent).
        return send_response(
            request,
            {"success": False, "message": str(exc), "data": None},
            status_code=500,
        )

    # Any HTTPException raised by a route — 5xx at error, 4xx at warning; both with traceback.
    @app.exception_handler(StarletteHTTPException)
    async def http_logging_handler(request: Request, exc: StarletteHTTPException) -> Response:
        log = logger.error if exc.status_code >= 500 else logger.warning
        log(
            "Request failed | request_id=%s | %d %s",
            get_request_id(request), exc.status_code, exc.detail, exc_info=exc,
        )
        response = await http_exception_handler(request, exc)  # keep the default shaping
        log_response(request, {"detail": exc.detail}, status_code=exc.status_code)
        return response

    # Request validation failure (422) → warning, with traceback.
    @app.exception_handler(RequestValidationError)
    async def validation_logging_handler(request: Request, exc: RequestValidationError) -> Response:
        logger.warning(
            "Request failed | request_id=%s | 422 validation error | %s",
            get_request_id(request), exc.errors(), exc_info=exc,
        )
        response = await request_validation_exception_handler(request, exc)  # keep the default shaping
        log_response(request, {"detail": exc.errors()}, status_code=422)
        return response

    # Catch-all: turn unhandled errors into a clean 500, logged with traceback.
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "Unhandled error | request_id=%s | %s: %s",
            get_request_id(request), type(exc).__name__, exc, exc_info=exc,
        )
        return send_response(
            request,
            {"success": False, "message": "internal error", "data": None},
            status_code=500,
        )

    # Register routers
    from app.api.routes_audio import router as audio_router
    from app.api.routes_health import router as health_router
    from app.api.routes_models import router as models_router
    from app.api.routes_openai import router as openai_router

    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(openai_router)
    app.include_router(audio_router)

    _install_custom_openapi(app)

    return app


def _install_custom_openapi(app: FastAPI) -> None:
    """
    Override app.openapi() to inject the chat-completion 200 examples.

    FastAPI serialises the schema with exclude_none=True, which strips the
    `x_metrics: null` key from the non-verbose example. We patch the examples
    in after generation so the `null` survives.
    """
    from fastapi.openapi.utils import get_openapi
    from app.api.routes_openai import CHAT_COMPLETION_200_EXAMPLES

    def custom_openapi() -> dict:
        if app.openapi_schema:
            return app.openapi_schema

        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
            tags=app.openapi_tags,
        )

        try:
            media = (
                schema["paths"]["/v1/chat/completions"]["post"]["responses"]["200"]
                ["content"]["application/json"]
            )
            media["examples"] = CHAT_COMPLETION_200_EXAMPLES
        except KeyError:
            pass

        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi


# The ASGI app instance used by uvicorn
app = create_app()
