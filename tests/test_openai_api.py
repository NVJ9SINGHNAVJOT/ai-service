"""
Automated tests for API endpoints and OpenAI-compatible request handling.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest


def test_health_endpoint(api_client):
    """GET /health should return 200 with status=ok."""
    resp = api_client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_models_list_endpoint(api_client):
    """GET /api/v1/models should return 200 with a list."""
    resp = api_client.get("/api/v1/models")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert isinstance(body["data"], list)


def test_models_list_endpoint_includes_input_modalities(api_client, monkeypatch):
    """GET /api/v1/models should expose input modality hints for clients."""
    from app.api import routes_models
    from app.schemas.model import ModelInfo, ModelSource, ModelState

    monkeypatch.setattr(
        routes_models._manager,
        "list_models",
        lambda: [
            ModelInfo(
                name="mlx-community__VisionModel",
                repo_id="mlx-community/VisionModel",
                source=ModelSource.downloaded,
                state=ModelState.ready,
                path="/tmp/fake-vision-model",
                loadable=True,
                input_modalities=["text", "image"],
                size_mb=123.45,
                created_at=None,
                updated_at=None,
            )
        ],
    )

    resp = api_client.get("/api/v1/models")

    assert resp.status_code == 200
    assert resp.json()["data"][0]["input_modalities"] == ["text", "image"]


def test_removed_model_delete_endpoint_returns_404(api_client):
    """DELETE /api/v1/models/{name} should no longer be mounted."""
    resp = api_client.delete("/api/v1/models/totally-fake-model-xyz")
    assert resp.status_code == 404


def test_removed_custom_inference_routes_return_404(api_client):
    """The legacy custom inference API should no longer be mounted."""
    resp = api_client.post(
        "/api/v1/inference/chat",
        json={"model": "my-model", "messages": [{"role": "user", "content": "Hello"}]},
    )

    assert resp.status_code == 404


def test_openai_chat_reuses_already_loaded_model(monkeypatch):
    """Chat completions should not reload a model when the same one is already loaded."""
    from app.main import app, inference_service
    from app.services.inference_service import InferenceService

    load_calls: list[tuple[object, str]] = []

    monkeypatch.setattr(InferenceService, "loaded_model_name", property(lambda self: "my-model"))
    monkeypatch.setattr(inference_service, "load", lambda model_path, model_name: load_calls.append((model_path, model_name)))
    monkeypatch.setattr(
        inference_service,
        "chat",
        lambda messages, max_tokens=None, temperature=None, top_p=None, repetition_penalty=None: ("ready", {}),
    )

    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json={"model": "my-model", "messages": [{"role": "user", "content": "Hello"}]})

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "ready"
    assert load_calls == []


def test_openai_chat_loads_requested_model_when_switching(monkeypatch):
    """Chat completions should load a new model when a different one is requested."""
    from app.api import routes_openai
    from app.main import app, inference_service
    from app.schemas.model import ModelInfo, ModelSource
    from app.services.inference_service import InferenceService

    load_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(InferenceService, "loaded_model_name", property(lambda self: "old-model"))
    monkeypatch.setattr(
        routes_openai._manager,
        "ensure_model_loadable",
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
    monkeypatch.setattr(inference_service, "load", lambda model_path, model_name: load_calls.append((str(model_path), model_name)))
    monkeypatch.setattr(
        inference_service,
        "chat",
        lambda messages, max_tokens=None, temperature=None, top_p=None, repetition_penalty=None: ("switched", {}),
    )

    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json={"model": "new-model", "messages": [{"role": "user", "content": "Hello"}]})

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "switched"
    assert load_calls == [("/tmp/new-model", "new-model")]


def test_models_unload_endpoint_unloads_current_model(monkeypatch):
    """The unload endpoint should call the shared inference service unload()."""
    from app.main import app, inference_service
    from app.services.inference_service import InferenceService

    monkeypatch.setattr(InferenceService, "loaded_model_name", property(lambda self: "my-model"))
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

    monkeypatch.setattr(InferenceService, "loaded_model_name", property(lambda self: "loaded-model"))

    with TestClient(app) as client:
        resp = client.post("/api/v1/models/unload", json={"name": "other-model"})

    assert resp.status_code == 400
    assert "not currently loaded" in resp.json()["detail"]


def test_openai_chat_completions_endpoint(api_client, monkeypatch):
    """POST /v1/chat/completions should return an OpenAI-compatible payload."""
    from app.api import routes_openai
    from app.main import inference_service
    from app.schemas.model import ModelInfo, ModelSource
    from app.services.inference_service import InferenceService

    monkeypatch.setattr(
        routes_openai._manager,
        "ensure_model_loadable",
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
        "chat",
        lambda messages, max_tokens=None, temperature=None, top_p=None, repetition_penalty=None: ("Hello from MLX", {}),
    )

    resp = api_client.post(
        "/v1/chat/completions",
        json={"model": "my-custom-model", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "my-custom-model"
    assert body["id"].startswith("chatcmpl-")
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == "Hello from MLX"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] >= body["usage"]["prompt_tokens"]
    assert body["x_metrics"] is None


def test_openai_chat_completions_accepts_developer_role(api_client, monkeypatch):
    """The OpenAI-compatible endpoint should accept developer messages."""
    from app.api import routes_openai
    from app.main import inference_service
    from app.schemas.model import ModelInfo, ModelSource
    from app.services.inference_service import InferenceService

    monkeypatch.setattr(
        routes_openai._manager,
        "ensure_model_loadable",
        lambda name: ModelInfo(
            name=name,
            repo_id=None,
            source=ModelSource.custom,
            path="/tmp/fake-model",
            loadable=True,
            input_modalities=["text"],
            size_mb=None,
            created_at=None,
            updated_at=None,
        ),
    )
    monkeypatch.setattr(inference_service, "load", lambda model_path, model_name: None)
    monkeypatch.setattr(InferenceService, "loaded_model_name", property(lambda self: None))
    monkeypatch.setattr(
        inference_service,
        "chat",
        lambda messages, max_tokens=None, temperature=None, top_p=None, repetition_penalty=None: ("Developer role works", {}),
    )

    resp = api_client.post(
        "/v1/chat/completions",
        json={
            "model": "my-custom-model",
            "messages": [{"role": "developer", "content": "You are terse."}, {"role": "user", "content": "Say hi"}],
        },
    )

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "Developer role works"


def test_openai_chat_completions_ignores_harmless_openai_fields(api_client, monkeypatch):
    """Safe OpenAI fields should be accepted even when unused locally."""
    from app.api import routes_openai
    from app.main import inference_service
    from app.schemas.model import ModelInfo, ModelSource
    from app.services.inference_service import InferenceService

    monkeypatch.setattr(
        routes_openai._manager,
        "ensure_model_loadable",
        lambda name: ModelInfo(
            name=name,
            repo_id=None,
            source=ModelSource.custom,
            path="/tmp/fake-model",
            loadable=True,
            input_modalities=["text"],
            size_mb=None,
            created_at=None,
            updated_at=None,
        ),
    )
    monkeypatch.setattr(inference_service, "load", lambda model_path, model_name: None)
    monkeypatch.setattr(InferenceService, "loaded_model_name", property(lambda self: None))
    monkeypatch.setattr(
        inference_service,
        "chat",
        lambda messages, max_tokens=None, temperature=None, top_p=None, repetition_penalty=None: ("Ignored extras okay", {}),
    )

    resp = api_client.post(
        "/v1/chat/completions",
        json={
            "model": "my-custom-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "store": False,
            "metadata": {"origin": "node-backend"},
            "service_tier": "default",
            "seed": 123,
            "safety_identifier": "user-1",
            "stream_options": {"include_usage": True},
            "n": 1,
            "max_completion_tokens": 32,
        },
    )

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "Ignored extras okay"


def test_openai_chat_completions_rejects_tools_with_400(api_client):
    """Unsupported advanced features should return a clear 400."""
    resp = api_client.post(
        "/v1/chat/completions",
        json={
            "model": "my-custom-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
        },
    )

    assert resp.status_code == 400
    assert "tools" in resp.json()["detail"]


def test_openai_chat_completions_rejects_unknown_extra_field_with_400(api_client):
    """Unknown OpenAI-ish extras should fail clearly instead of being silently misread."""
    resp = api_client.post(
        "/v1/chat/completions",
        json={
            "model": "my-custom-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "totally_unknown_option": True,
        },
    )

    assert resp.status_code == 400
    assert "totally_unknown_option" in resp.json()["detail"]


def test_openai_chat_completions_rejects_n_greater_than_one(api_client):
    """We currently support only one completion choice per request."""
    resp = api_client.post(
        "/v1/chat/completions",
        json={"model": "my-custom-model", "messages": [{"role": "user", "content": "Hello"}], "n": 2},
    )

    assert resp.status_code == 400
    assert "supported only with the value 1" in resp.json()["detail"]


def test_openai_chat_completions_supports_stop_in_non_streaming(api_client, monkeypatch):
    """The endpoint should stop generation at the requested stop sequence."""
    from app.api import routes_openai
    from app.main import inference_service
    from app.schemas.model import ModelInfo, ModelSource
    from app.services.inference_service import InferenceService

    monkeypatch.setattr(
        routes_openai._manager,
        "ensure_model_loadable",
        lambda name: ModelInfo(
            name=name,
            repo_id=None,
            source=ModelSource.custom,
            path="/tmp/fake-model",
            loadable=True,
            input_modalities=["text"],
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
            [("Hello END world", {"finish_reason": "length", "prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6})]
        ),
    )

    resp = api_client.post(
        "/v1/chat/completions",
        json={"model": "my-custom-model", "messages": [{"role": "user", "content": "Hello"}], "stop": "END"},
    )

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "Hello "


def test_openai_chat_completions_supports_stop_in_streaming(api_client, monkeypatch):
    """Streaming should trim at stop sequences even when they span chunks."""
    from app.api import routes_openai
    from app.main import inference_service
    from app.schemas.model import ModelInfo, ModelSource
    from app.services.inference_service import InferenceService

    monkeypatch.setattr(
        routes_openai._manager,
        "ensure_model_loadable",
        lambda name: ModelInfo(
            name=name,
            repo_id=None,
            source=ModelSource.custom,
            path="/tmp/fake-model",
            loadable=True,
            input_modalities=["text"],
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
            [("Hello E", None), ("ND world", {"finish_reason": "length", "prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6})]
        ),
    )

    with api_client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "my-custom-model", "messages": [{"role": "user", "content": "Hello"}], "stream": True, "stop": "END"},
    ) as resp:
        body = "".join(resp.iter_text())

    assert resp.status_code == 200
    assert '"delta": {"content": "Hello"}' in body
    assert '"delta": {"content": " "}' in body
    assert "END world" not in body
    assert "data: [DONE]" in body


def test_openai_chat_completions_streaming(api_client, monkeypatch):
    """POST /v1/chat/completions should stream OpenAI-style SSE chunks."""
    from app.api import routes_openai
    from app.main import inference_service
    from app.schemas.model import ModelInfo, ModelSource
    from app.services.inference_service import InferenceService

    monkeypatch.setattr(
        routes_openai._manager,
        "ensure_model_loadable",
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
                (
                    " world",
                    {
                        "prompt_tokens": 2,
                        "completion_tokens": 2,
                        "total_tokens": 4,
                        "finish_reason": "stop",
                        "metrics": {
                            "total_duration_s": 1.25,
                            "prompt_eval_duration_s": 0.25,
                            "prompt_eval_rate": 8.0,
                            "eval_duration_s": 1.0,
                            "eval_rate": 2.0,
                        },
                    },
                ),
            ]
        ),
    )
    inference_service._last_load_duration_s = 0.5

    with api_client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "my-custom-model", "messages": [{"role": "user", "content": "Hello"}], "stream": True, "verbose": True},
    ) as resp:
        body = "".join(resp.iter_text())

    assert resp.status_code == 200
    assert 'data: {"id": "chatcmpl-' in body
    assert '"object": "chat.completion.chunk"' in body
    assert '"delta": {"role": "assistant"}' in body
    assert '"delta": {"content": "Hello"}' in body
    assert '"finish_reason": "stop"' in body
    assert "data: [DONE]" in body
    assert '"x_metrics": {"total_duration_s": 1.25, "load_duration_s": 0.5' in body


def test_openai_chat_completions_streaming_include_usage_without_verbose(api_client, monkeypatch):
    """stream_options.include_usage should emit a usage chunk even when verbose is off (no x_metrics)."""
    from app.api import routes_openai
    from app.main import inference_service
    from app.schemas.model import ModelInfo, ModelSource
    from app.services.inference_service import InferenceService

    monkeypatch.setattr(
        routes_openai._manager,
        "ensure_model_loadable",
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
                (
                    " world",
                    {
                        "prompt_tokens": 8,
                        "completion_tokens": 18,
                        "total_tokens": 26,
                        "finish_reason": "stop",
                    },
                ),
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
            "stream_options": {"include_usage": True},
        },
    ) as resp:
        body = "".join(resp.iter_text())

    assert resp.status_code == 200
    # Usage chunk present with the token counts; no x_metrics since verbose was off.
    assert '"usage": {"prompt_tokens": 8, "completion_tokens": 18, "total_tokens": 26}' in body
    assert "x_metrics" not in body
    assert "data: [DONE]" in body


def test_openai_chat_completions_verbose_non_streaming(api_client, monkeypatch):
    """verbose=true should include timing metrics in the OpenAI response."""
    from app.api import routes_openai
    from app.main import inference_service
    from app.schemas.model import ModelInfo, ModelSource
    from app.services.inference_service import InferenceService

    monkeypatch.setattr(
        routes_openai._manager,
        "ensure_model_loadable",
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
                (
                    " world",
                    {
                        "prompt_tokens": 13,
                        "completion_tokens": 43,
                        "total_tokens": 56,
                        "finish_reason": "stop",
                        "metrics": {
                            "total_duration_s": 1.256830584,
                            "prompt_eval_duration_s": 0.267349292,
                            "prompt_eval_rate": 48.63,
                            "eval_duration_s": 0.890170291,
                            "eval_rate": 48.31,
                        },
                    },
                ),
            ]
        ),
    )
    inference_service._last_load_duration_s = 0.094960042

    resp = api_client.post(
        "/v1/chat/completions",
        json={"model": "my-custom-model", "messages": [{"role": "user", "content": "Hello"}], "verbose": True},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "Hello world"
    assert body["usage"]["prompt_tokens"] == 13
    assert body["usage"]["completion_tokens"] == 43
    assert body["x_metrics"]["total_duration_s"] == 1.256830584
    assert body["x_metrics"]["load_duration_s"] == 0.094960042
    assert body["x_metrics"]["prompt_eval_count"] == 13
    assert body["x_metrics"]["eval_rate"] == 48.31


def test_openai_chat_completions_verbose_warm_turn_omits_load_duration(api_client, monkeypatch):
    """On a warm turn (model already resident) load_duration_s is omitted; other metrics remain."""
    from app.api import routes_openai
    from app.main import inference_service
    from app.schemas.model import ModelInfo, ModelSource
    from app.services.inference_service import InferenceService

    monkeypatch.setattr(
        routes_openai._manager,
        "ensure_model_loadable",
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
    # Model is already the requested one — this turn loads nothing.
    monkeypatch.setattr(InferenceService, "loaded_model_name", property(lambda self: "my-custom-model"))
    monkeypatch.setattr(
        inference_service,
        "chat_stream",
        lambda messages, max_tokens=None, temperature=None, top_p=None, repetition_penalty=None: iter(
            [
                ("Hello", None),
                (
                    " world",
                    {
                        "prompt_tokens": 13,
                        "completion_tokens": 43,
                        "total_tokens": 56,
                        "finish_reason": "stop",
                        "metrics": {
                            "total_duration_s": 1.256830584,
                            "prompt_eval_duration_s": 0.267349292,
                            "prompt_eval_rate": 48.63,
                            "eval_duration_s": 0.890170291,
                            "eval_rate": 48.31,
                        },
                    },
                ),
            ]
        ),
    )
    # A stale value from a prior load — must NOT be reported on this warm turn.
    inference_service._last_load_duration_s = 0.094960042

    resp = api_client.post(
        "/v1/chat/completions",
        json={"model": "my-custom-model", "messages": [{"role": "user", "content": "Hello"}], "verbose": True},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["x_metrics"]["load_duration_s"] is None
    assert body["x_metrics"]["total_duration_s"] == 1.256830584
    assert body["x_metrics"]["eval_rate"] == 48.31


def test_openai_chat_completions_endpoint_accepts_multimodal_messages(api_client, monkeypatch):
    """POST /v1/chat/completions should route image requests through mlx-vlm."""
    from app.api import routes_openai
    from app.main import inference_service, media_inference_service
    from app.schemas.model import ModelInfo, ModelSource
    from app.services.media_inference_service import MediaInferenceService

    def should_not_use_text_loader(name):
        raise AssertionError("Text loader should not be used for image requests.")

    monkeypatch.setattr(routes_openai._manager, "ensure_model_loadable", should_not_use_text_loader)
    monkeypatch.setattr(
        routes_openai._manager,
        "ensure_model_files_ready",
        lambda name: ModelInfo(
            name=name,
            repo_id=None,
            source=ModelSource.custom,
            path="/tmp/fake-vision-model",
            loadable=True,
            size_mb=None,
            created_at=None,
            updated_at=None,
        ),
    )
    monkeypatch.setattr(inference_service, "unload", lambda: None)
    monkeypatch.setattr(media_inference_service, "load", lambda model_path, model_name: None)
    monkeypatch.setattr(MediaInferenceService, "loaded_model_name", property(lambda self: None))
    monkeypatch.setattr(
        media_inference_service,
        "chat",
        lambda messages, max_tokens=None, temperature=None, top_p=None, repetition_penalty=None: (
            "Vision response",
            {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
        ),
    )

    resp = api_client.post(
        "/v1/chat/completions",
        json={
            "model": "my-vision-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {"type": "image_url", "image_url": {"url": "/tmp/example.jpg"}},
                    ],
                }
            ],
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "Vision response"
    assert body["usage"]["prompt_tokens"] == 7
    assert body["usage"]["completion_tokens"] == 2


def test_openai_chat_completions_streaming_accepts_input_image_parts(api_client, monkeypatch):
    """Streaming multimodal requests should emit SSE chunks through the media service."""
    from app.api import routes_openai
    from app.main import inference_service, media_inference_service
    from app.schemas.model import ModelInfo, ModelSource
    from app.services.media_inference_service import MediaInferenceService

    def should_not_use_text_loader(name):
        raise AssertionError("Text loader should not be used for image requests.")

    monkeypatch.setattr(routes_openai._manager, "ensure_model_loadable", should_not_use_text_loader)
    monkeypatch.setattr(
        routes_openai._manager,
        "ensure_model_files_ready",
        lambda name: ModelInfo(
            name=name,
            repo_id=None,
            source=ModelSource.custom,
            path="/tmp/fake-vision-model",
            loadable=True,
            size_mb=None,
            created_at=None,
            updated_at=None,
        ),
    )
    monkeypatch.setattr(inference_service, "unload", lambda: None)
    monkeypatch.setattr(media_inference_service, "load", lambda model_path, model_name: None)
    monkeypatch.setattr(MediaInferenceService, "loaded_model_name", property(lambda self: None))
    monkeypatch.setattr(
        media_inference_service,
        "chat_stream",
        lambda messages, max_tokens=None, temperature=None, top_p=None, repetition_penalty=None: iter(
            [
                ("Vision", None),
                (
                    " stream",
                    {
                        "prompt_tokens": 5,
                        "completion_tokens": 2,
                        "total_tokens": 7,
                        "finish_reason": "stop",
                        "metrics": {
                            "total_duration_s": 0.9,
                            "prompt_eval_duration_s": 0.2,
                            "prompt_eval_rate": 25.0,
                            "eval_duration_s": 0.7,
                            "eval_rate": 2.86,
                        },
                    },
                ),
            ]
        ),
    )
    media_inference_service._last_load_duration_s = 0.3

    with api_client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "my-vision-model",
            "stream": True,
            "verbose": True,
            "messages": [{"role": "user", "content": [{"type": "input_text", "text": "What do you see?"}, {"type": "input_image", "image_url": "/tmp/example.jpg"}]}],
        },
    ) as resp:
        body = "".join(resp.iter_text())

    assert resp.status_code == 200
    assert '"delta": {"content": "Vision"}' in body
    assert '"delta": {"content": " stream"}' in body
    assert '"x_metrics": {"total_duration_s": 0.9, "load_duration_s": 0.3' in body
    assert "data: [DONE]" in body


def test_openai_chat_completions_endpoint_accepts_input_audio_parts(api_client, monkeypatch):
    """Audio-bearing requests should use the mlx-vlm service for chat completions."""
    from app.api import routes_openai
    from app.main import inference_service, media_inference_service
    from app.schemas.model import ModelInfo, ModelSource
    from app.services.media_inference_service import MediaInferenceService

    def should_not_use_text_loader(name):
        raise AssertionError("Text loader should not be used for audio requests.")

    monkeypatch.setattr(routes_openai._manager, "ensure_model_loadable", should_not_use_text_loader)
    monkeypatch.setattr(
        routes_openai._manager,
        "ensure_model_files_ready",
        lambda name: ModelInfo(
            name=name,
            repo_id=None,
            source=ModelSource.custom,
            path="/tmp/fake-audio-model",
            loadable=True,
            input_modalities=["text", "audio"],
            size_mb=None,
            created_at=None,
            updated_at=None,
        ),
    )
    monkeypatch.setattr(inference_service, "unload", lambda: None)
    monkeypatch.setattr(media_inference_service, "load", lambda model_path, model_name: None)
    monkeypatch.setattr(MediaInferenceService, "loaded_model_name", property(lambda self: None))
    monkeypatch.setattr(
        media_inference_service,
        "chat",
        lambda messages, max_tokens=None, temperature=None, top_p=None, repetition_penalty=None: (
            "Audio response",
            {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        ),
    )

    resp = api_client.post(
        "/v1/chat/completions",
        json={
            "model": "my-audio-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Transcribe this clip"},
                        {"type": "input_audio", "input_audio": {"data": "base64-audio-data", "format": "wav"}},
                    ],
                }
            ],
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "Audio response"
    assert body["usage"]["total_tokens"] == 6


@pytest.mark.anyio
async def test_openai_streaming_stops_when_client_disconnects():
    """The streaming route should stop yielding chunks once the client disconnects."""
    from app.api import routes_openai

    class FakeRequest:
        def __init__(self):
            self.calls = 0

        async def is_disconnected(self):
            self.calls += 1
            return self.calls >= 2

    request = FakeRequest()

    async def collect():
        yielded = []
        if await request.is_disconnected():
            return yielded
        for chunk in routes_openai._stream_with_stop_sequences(iter([("Hello", None), (" world", None)]), []):
            if await request.is_disconnected():
                return yielded
            yielded.append(chunk)
            if await request.is_disconnected():
                return yielded
        return yielded

    assert await collect() == []
