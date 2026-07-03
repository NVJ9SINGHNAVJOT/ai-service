from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient


@pytest.fixture()
def correlation_client() -> TestClient:
    """Minimal app wrapped in LoggingMiddleware that echoes the ids it recorded."""
    from app.api.middleware import LoggingMiddleware

    app = FastAPI()
    app.add_middleware(LoggingMiddleware)

    @app.get("/probe")
    async def probe(request: Request):
        return {
            "correlation_id": request.state.correlation_id,
            "request_id": request.state.request_id,
        }

    return TestClient(app)


def test_correlation_id_taken_from_incoming_header(correlation_client: TestClient):
    """The X-Correlation-ID forwarded by the caller is recorded on request.state."""
    resp = correlation_client.get("/probe", headers={"X-Correlation-ID": "central-123"})

    body = resp.json()
    assert body["correlation_id"] == "central-123"
    # request_id is generated locally and is independent of the correlation id.
    assert body["request_id"] not in ("central-123", "unknown")


def test_correlation_id_defaults_to_unknown_when_header_absent(correlation_client: TestClient):
    """A request without the header still logs cleanly with a sentinel correlation id."""
    resp = correlation_client.get("/probe")

    assert resp.json()["correlation_id"] == "unknown"


def test_setup_logging_quiets_noisy_dependency_loggers():
    """Network-heavy library INFO logs should not disrupt CLI progress output."""
    from app.core.logging import setup_logging

    setup_logging()

    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING
    assert logging.getLogger("huggingface_hub").level == logging.WARNING


def test_setup_logging_keeps_app_logging_level_configurable():
    """Application loggers should still follow the configured root level."""
    from app.core.logging import setup_logging

    setup_logging(level=logging.DEBUG)

    assert logging.getLogger().level == logging.DEBUG
