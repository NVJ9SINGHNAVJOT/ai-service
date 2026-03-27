"""
Focused tests for API model lifecycle behavior.

These cover the in-memory rules that matter for request flow:
- reuse the already loaded model for repeated requests
- load/swap when a different model is requested
- unload only when the unload endpoint is called
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_generate_reuses_already_loaded_model(monkeypatch):
    """Inference should not reload a model when the same one is already loaded."""
    from app.main import app, inference_service
    from app.services.inference_service import InferenceService

    load_calls: list[tuple[object, str]] = []

    monkeypatch.setattr(
        InferenceService,
        "loaded_model_name",
        property(lambda self: "my-model"),
    )
    monkeypatch.setattr(
        inference_service,
        "load",
        lambda model_path, model_name: load_calls.append((model_path, model_name)),
    )
    monkeypatch.setattr(
        inference_service,
        "generate",
        lambda prompt, max_tokens=None, temperature=None, top_p=None, repetition_penalty=None: ("ready", {}),
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/inference/generate",
            json={"model": "my-model", "prompt": "Hello"},
        )

    assert resp.status_code == 200
    assert resp.json()["data"]["text"] == "ready"
    assert load_calls == []


def test_generate_loads_requested_model_when_switching(monkeypatch):
    """Inference should load a new model when a different one is requested."""
    from app.api import routes_inference
    from app.main import app, inference_service
    from app.schemas.model import ModelInfo, ModelSource
    from app.services.inference_service import InferenceService

    load_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        InferenceService,
        "loaded_model_name",
        property(lambda self: "old-model"),
    )
    monkeypatch.setattr(
        routes_inference._manager,
        "get_model",
        lambda name: ModelInfo(
            name=name,
            repo_id=None,
            source=ModelSource.custom,
            path=f"/tmp/{name}",
            loadable=True,
            size_mb=None,
            created_at=None,
            updated_at=None,
        ),
    )
    monkeypatch.setattr(
        inference_service,
        "load",
        lambda model_path, model_name: load_calls.append((str(model_path), model_name)),
    )
    monkeypatch.setattr(
        inference_service,
        "generate",
        lambda prompt, max_tokens=None, temperature=None, top_p=None, repetition_penalty=None: ("switched", {}),
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/inference/generate",
            json={"model": "new-model", "prompt": "Hello"},
        )

    assert resp.status_code == 200
    assert resp.json()["data"]["text"] == "switched"
    assert load_calls == [("/tmp/new-model", "new-model")]


def test_models_unload_endpoint_unloads_current_model(monkeypatch):
    """The unload endpoint should call the shared inference service unload()."""
    from app.main import app, inference_service
    from app.services.inference_service import InferenceService

    monkeypatch.setattr(
        InferenceService,
        "loaded_model_name",
        property(lambda self: "my-model"),
    )
    monkeypatch.setattr(inference_service, "unload", lambda: "my-model")

    with TestClient(app) as client:
        resp = client.post("/api/v1/models/unload", json={"name": "my-model"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["unloaded_model"] == "my-model"


def test_models_unload_endpoint_rejects_mismatched_name(monkeypatch):
    """The unload endpoint should reject requests for a different loaded model."""
    from app.main import app
    from app.services.inference_service import InferenceService

    monkeypatch.setattr(
        InferenceService,
        "loaded_model_name",
        property(lambda self: "loaded-model"),
    )

    with TestClient(app) as client:
        resp = client.post("/api/v1/models/unload", json={"name": "other-model"})

    assert resp.status_code == 400
    assert "not currently loaded" in resp.json()["detail"]
