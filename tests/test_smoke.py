"""
Smoke tests — fast, no model downloads required.

These tests verify:
1. Configuration loads without error.
2. ModelManager can be instantiated and directories are created.
3. The FastAPI app starts and /health responds.
4. Model name sanitisation is correct.
5. Path traversal is blocked.
6. An empty model list is returned correctly.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_models_dir(tmp_path: Path):
    """Provide a temporary models directory tree."""
    downloaded = tmp_path / "downloaded"
    custom = tmp_path / "custom"
    downloaded.mkdir()
    custom.mkdir()
    return tmp_path


@pytest.fixture()
def manager(tmp_models_dir: Path, monkeypatch):
    """ModelManager pointing at a temporary directory."""
    from app.config import Settings
    from app.services.model_manager import ModelManager

    cfg = Settings(
        downloaded_models_dir=str(tmp_models_dir / "downloaded"),
        custom_models_dir=str(tmp_models_dir / "custom"),
        model_registry_file=str(tmp_models_dir / "registry.json"),
    )
    return ModelManager(cfg=cfg)


@pytest.fixture()
def api_client():
    """TestClient for the FastAPI application."""
    from app.main import app
    return TestClient(app)


# ── Configuration ────────────────────────────────────────────────────────────

def test_settings_load():
    """Settings object should initialise without error."""
    from app.config import settings
    assert settings.api_port > 0
    assert settings.default_max_tokens > 0


# ── Model name sanitisation ───────────────────────────────────────────────────

def test_sanitize_repo_id_basic():
    from app.services.model_manager import _sanitize_repo_id
    assert _sanitize_repo_id("mlx-community/Llama-3.2-3B-Instruct-4bit") == \
        "mlx-community__Llama-3.2-3B-Instruct-4bit"


def test_sanitize_repo_id_spaces():
    from app.services.model_manager import _sanitize_repo_id
    result = _sanitize_repo_id("org/model name")
    assert "/" not in result
    assert " " not in result


# ── ModelManager — basic operations ──────────────────────────────────────────

def test_list_models_empty(manager):
    """An empty models directory should return an empty list."""
    models = manager.list_models()
    assert models == []


def test_list_models_custom(manager, tmp_models_dir):
    """A directory placed in custom/ should appear in list_models()."""
    fake_model = tmp_models_dir / "custom" / "my-custom-model"
    fake_model.mkdir()
    # Add minimal indicator files so loadable=True
    (fake_model / "config.json").write_text("{}")
    (fake_model / "tokenizer_config.json").write_text("{}")

    models = manager.list_models()
    assert len(models) == 1
    assert models[0].name == "my-custom-model"
    assert models[0].source.value == "custom"
    assert models[0].loadable is True


def test_list_models_downloaded(manager, tmp_models_dir):
    """A directory placed in downloaded/ + registry entry should appear correctly."""
    fake_model = tmp_models_dir / "downloaded" / "mlx-community__TestModel"
    fake_model.mkdir()
    (fake_model / "config.json").write_text("{}")
    (fake_model / "tokenizer_config.json").write_text("{}")

    # Write a registry entry manually
    registry = {
        "mlx-community__TestModel": {
            "name": "mlx-community__TestModel",
            "repo_id": "mlx-community/TestModel",
            "path": str(fake_model),
            "source": "downloaded",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
        }
    }
    (tmp_models_dir / "registry.json").write_text(json.dumps(registry))

    models = manager.list_models()
    assert any(m.name == "mlx-community__TestModel" for m in models)
    m = next(m for m in models if m.name == "mlx-community__TestModel")
    assert m.repo_id == "mlx-community/TestModel"
    assert m.loadable is True


def test_get_model_not_found(manager):
    """get_model() should raise ModelNotFoundError for unknown models."""
    from app.core.exceptions import ModelNotFoundError
    with pytest.raises(ModelNotFoundError):
        manager.get_model("nonexistent-model")


def test_delete_model(manager, tmp_models_dir):
    """Delete should remove the directory and unregister the model."""
    fake = tmp_models_dir / "downloaded" / "mlx-community__DeleteMe"
    fake.mkdir()
    (fake / "config.json").write_text("{}")
    (fake / "tokenizer_config.json").write_text("{}")

    registry = {
        "mlx-community__DeleteMe": {
            "name": "mlx-community__DeleteMe",
            "repo_id": "mlx-community/DeleteMe",
            "path": str(fake),
            "source": "downloaded",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
        }
    }
    (tmp_models_dir / "registry.json").write_text(json.dumps(registry))

    manager.delete("mlx-community__DeleteMe")
    assert not fake.exists()

    remaining = manager.list_models()
    assert not any(m.name == "mlx-community__DeleteMe" for m in remaining)


def test_delete_custom_blocked_by_default(manager, tmp_models_dir):
    """Deleting a custom model without allow_custom should raise InvalidModelPathError."""
    from app.core.exceptions import InvalidModelPathError

    custom = tmp_models_dir / "custom" / "my-custom"
    custom.mkdir()
    (custom / "config.json").write_text("{}")
    (custom / "tokenizer_config.json").write_text("{}")

    with pytest.raises(InvalidModelPathError):
        manager.delete("my-custom", allow_custom=False)


def test_delete_custom_allowed(manager, tmp_models_dir):
    """Deleting a custom model with allow_custom=True should succeed."""
    custom = tmp_models_dir / "custom" / "my-custom"
    custom.mkdir()
    (custom / "config.json").write_text("{}")
    (custom / "tokenizer_config.json").write_text("{}")

    manager.delete("my-custom", allow_custom=True)
    assert not custom.exists()


# ── Path traversal ────────────────────────────────────────────────────────────

def test_path_traversal_blocked(manager):
    """_safe_downloaded_path() must reject traversal attempts."""
    from app.core.exceptions import InvalidModelPathError
    with pytest.raises(InvalidModelPathError):
        manager._safe_downloaded_path("../../etc/passwd")


# ── FastAPI health ────────────────────────────────────────────────────────────

def test_health_endpoint(api_client):
    """GET /health should return 200 with status=ok."""
    resp = api_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_models_list_endpoint(api_client):
    """GET /api/v1/models should return 200 with a list."""
    resp = api_client.get("/api/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert isinstance(data["data"], list)


def test_delete_nonexistent_model_returns_404(api_client):
    """DELETE /api/v1/models/nonexistent should return 404."""
    resp = api_client.delete("/api/v1/models/totally-fake-model-xyz")
    assert resp.status_code == 404


def test_openai_chat_completions_endpoint(api_client, monkeypatch):
    """POST /v1/chat/completions should return an OpenAI-compatible payload."""
    from app.api import routes_openai
    from app.schemas.model import ModelInfo, ModelSource
    from app.main import inference_service
    from app.services.inference_service import InferenceService

    monkeypatch.setattr(
        routes_openai._manager,
        "get_model",
        lambda name: ModelInfo(
            name=name,
            repo_id=None,
            source=ModelSource.custom,
            path="/tmp/fake-model",
            loadable=True,
            size_mb=None,
            created_at=None,
            updated_at=None,
        ),
    )

    monkeypatch.setattr(inference_service, "load", lambda model_path, model_name: None)
    monkeypatch.setattr(
        InferenceService,
        "loaded_model_name",
        property(lambda self: None),
    )
    monkeypatch.setattr(
        inference_service,
        "chat",
        lambda messages, max_tokens=None, temperature=None, top_p=None: ("Hello from MLX", {}),
    )

    resp = api_client.post(
        "/v1/chat/completions",
        json={
            "model": "my-custom-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 32,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "my-custom-model"
    assert data["id"].startswith("chatcmpl-")
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == "Hello from MLX"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"]["total_tokens"] >= data["usage"]["prompt_tokens"]


def test_openai_chat_completions_streaming(api_client, monkeypatch):
    """POST /v1/chat/completions should stream OpenAI-style SSE chunks."""
    from app.api import routes_openai
    from app.schemas.model import ModelInfo, ModelSource
    from app.main import inference_service
    from app.services.inference_service import InferenceService

    monkeypatch.setattr(
        routes_openai._manager,
        "get_model",
        lambda name: ModelInfo(
            name=name,
            repo_id=None,
            source=ModelSource.custom,
            path="/tmp/fake-model",
            loadable=True,
            size_mb=None,
            created_at=None,
            updated_at=None,
        ),
    )
    monkeypatch.setattr(inference_service, "load", lambda model_path, model_name: None)
    monkeypatch.setattr(InferenceService, "loaded_model_name", property(lambda self: None))
    monkeypatch.setattr(
        inference_service,
        "chat_stream",
        lambda messages, max_tokens=None, temperature=None, top_p=None, repetition_penalty=None: iter(
            [
                ("Hello", None),
                (" world", {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4, "finish_reason": "stop"}),
            ]
        ),
    )

    with api_client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "my-custom-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    ) as resp:
        body = "".join(resp.iter_text())

    assert resp.status_code == 200
    assert 'data: {"id": "chatcmpl-' in body
    assert '"object": "chat.completion.chunk"' in body
    assert '"delta": {"role": "assistant"}' in body
    assert '"delta": {"content": "Hello"}' in body
    assert '"finish_reason": "stop"' in body
    assert "data: [DONE]" in body


def test_custom_chat_streaming_endpoint(api_client, monkeypatch):
    """POST /api/v1/inference/chat should stream newline-delimited JSON chunks."""
    from app.api import routes_inference
    from app.schemas.model import ModelInfo, ModelSource
    from app.main import inference_service
    from app.services.inference_service import InferenceService

    monkeypatch.setattr(
        routes_inference._manager,
        "get_model",
        lambda name: ModelInfo(
            name=name,
            repo_id=None,
            source=ModelSource.custom,
            path="/tmp/fake-model",
            loadable=True,
            size_mb=None,
            created_at=None,
            updated_at=None,
        ),
    )
    monkeypatch.setattr(inference_service, "load", lambda model_path, model_name: None)
    monkeypatch.setattr(InferenceService, "loaded_model_name", property(lambda self: None))
    monkeypatch.setattr(
        inference_service,
        "chat_stream",
        lambda messages, max_tokens=None, temperature=None, top_p=None, repetition_penalty=None: iter(
            [
                ("Hello", None),
                (" world", {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4, "finish_reason": "stop"}),
            ]
        ),
    )

    with api_client.stream(
        "POST",
        "/api/v1/inference/chat",
        json={
            "model": "my-custom-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    ) as resp:
        lines = [line for line in resp.iter_lines() if line]

    assert resp.status_code == 200
    chunks = [json.loads(line) for line in lines]
    assert chunks[0]["text"] == "Hello"
    assert chunks[0]["done"] is False
    assert chunks[-1]["text"] == " world"
    assert chunks[-1]["done"] is True
    assert chunks[-1]["usage"]["finish_reason"] == "stop"


def test_chat_session_uses_streaming(monkeypatch):
    """ChatSession should build the assistant response from streamed chunks."""
    from app.schemas.inference import ChatMessage, Role
    from app.services.chat_session import ChatSession

    prompts = iter(["Hello", "quit"])
    printed: list[str] = []

    monkeypatch.setattr("app.services.chat_session.Prompt.ask", lambda _: next(prompts))
    monkeypatch.setattr("app.services.chat_session.console.print", lambda *args, **kwargs: printed.append("" if not args else str(args[0])))

    session = ChatSession(model_path=Path("/tmp/fake-model"), model_name="my-model")
    session._history = [ChatMessage(role=Role.system, content="You are helpful.")]
    monkeypatch.setattr(
        session._svc,
        "chat_stream",
        lambda messages, max_tokens=None, temperature=None, top_p=None, repetition_penalty=None: iter(
            [("Hello", None), (" world", {"finish_reason": "stop"})]
        ),
    )

    session._loop()

    assert session._history[-1].role == Role.assistant
    assert session._history[-1].content == "Hello world"
