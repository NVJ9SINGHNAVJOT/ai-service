"""
Automated tests for model discovery, doctor output, and model-manager behavior.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest


def test_settings_load():
    """Settings object should initialise without error."""
    from app.config import settings

    assert settings.api_port > 0
    assert settings.default_max_tokens > 0


def test_sanitize_repo_id_basic():
    from app.services.model_manager import _sanitize_repo_id

    assert _sanitize_repo_id("mlx-community/Llama-3.2-3B-Instruct-4bit") == "mlx-community__Llama-3.2-3B-Instruct-4bit"


def test_sanitize_repo_id_spaces():
    from app.services.model_manager import _sanitize_repo_id

    result = _sanitize_repo_id("org/model name")
    assert "/" not in result
    assert " " not in result


def test_list_models_empty(manager):
    """An empty models directory should return an empty list."""
    assert manager.list_models() == []


def test_list_models_custom(manager, tmp_models_dir):
    """A directory placed in custom/ should appear in list_models()."""
    fake_model = tmp_models_dir / "custom" / "my-custom-model"
    fake_model.mkdir()
    (fake_model / "config.json").write_text("{}")
    (fake_model / "tokenizer_config.json").write_text("{}")

    models = manager.list_models()

    assert len(models) == 1
    assert models[0].name == "my-custom-model"
    assert models[0].source.value == "custom"
    assert models[0].loadable is True


def test_list_models_downloaded(manager, tmp_models_dir):
    """A directory placed in downloaded/ with registry data should list correctly."""
    fake_model = tmp_models_dir / "downloaded" / "mlx-community__TestModel"
    fake_model.mkdir()
    (fake_model / "config.json").write_text("{}")
    (fake_model / "tokenizer_config.json").write_text("{}")

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
    model = next(m for m in models if m.name == "mlx-community__TestModel")

    assert model.repo_id == "mlx-community/TestModel"
    assert model.loadable is True
    assert model.state.value == "ready"


def test_diagnose_model_reports_ready(manager, tmp_models_dir):
    """Doctor should report a healthy model as ready to load."""
    fake_model = tmp_models_dir / "downloaded" / "mlx-community__DoctorReady"
    fake_model.mkdir()
    (fake_model / "config.json").write_text('{"model_type":"gemma"}')
    (fake_model / "tokenizer_config.json").write_text("{}")

    registry = {
        "mlx-community__DoctorReady": {
            "name": "mlx-community__DoctorReady",
            "repo_id": "mlx-community/DoctorReady",
            "path": str(fake_model),
            "source": "downloaded",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
        }
    }
    (tmp_models_dir / "registry.json").write_text(json.dumps(registry))

    diagnosis = manager.diagnose_model("mlx-community__DoctorReady")

    assert diagnosis.model_type == "gemma"
    assert diagnosis.loadable is True
    assert diagnosis.input_modalities == ["text"]
    assert diagnosis.summary == "Model looks ready to load."
    assert diagnosis.recommendations


def test_diagnose_model_reports_multimodal_inputs(manager, tmp_models_dir):
    """Doctor should infer image and audio inputs from config hints."""
    fake_model = tmp_models_dir / "downloaded" / "mlx-community__DoctorMultimodal"
    fake_model.mkdir()
    (fake_model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_omni_moe",
                "vision_config": {"image_size": 448},
                "audio_config": {"sampling_rate": 16000},
            }
        )
    )
    (fake_model / "tokenizer_config.json").write_text("{}")

    registry = {
        "mlx-community__DoctorMultimodal": {
            "name": "mlx-community__DoctorMultimodal",
            "repo_id": "mlx-community/DoctorMultimodal",
            "path": str(fake_model),
            "source": "downloaded",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
        }
    }
    (tmp_models_dir / "registry.json").write_text(json.dumps(registry))

    diagnosis = manager.diagnose_model("mlx-community__DoctorMultimodal")

    assert diagnosis.input_modalities == ["text", "image", "audio"]


def test_list_models_includes_downloading_runtime_state(manager):
    """A model being downloaded should appear with state=downloading."""
    from app.services.model_runtime_state import ModelRuntimeState

    fake_model = manager._cfg.downloaded_models_path / "mlx-community__Gemma-4-e4b-it-bf16"
    fake_model.mkdir()
    (fake_model / "config.json").write_text("{}")
    (fake_model / "tokenizer_config.json").write_text("{}")

    runtime_state = ModelRuntimeState(cfg=manager._cfg)
    marker = runtime_state.mark_downloading(
        "mlx-community__Gemma-4-e4b-it-bf16",
        repo_id="mlx-community/gemma-4-e4b-it-bf16",
    )
    try:
        models = manager.list_models()
    finally:
        runtime_state.clear_marker(marker)

    model = next(m for m in models if m.name == "mlx-community__Gemma-4-e4b-it-bf16")
    assert model.source.value == "downloaded"
    assert model.state.value == "downloading"
    assert model.loadable is False
    assert model.repo_id == "mlx-community/gemma-4-e4b-it-bf16"


def test_list_models_marks_loaded_model_as_running(manager, tmp_models_dir):
    """A model with a live usage marker should appear with state=running."""
    from app.services.model_runtime_state import ModelRuntimeState

    fake_model = tmp_models_dir / "downloaded" / "mlx-community__Qwen3.5-35B-A3B-4bit"
    fake_model.mkdir()
    (fake_model / "config.json").write_text("{}")
    (fake_model / "tokenizer_config.json").write_text("{}")

    runtime_state = ModelRuntimeState(cfg=manager._cfg)
    marker = runtime_state.mark_running("mlx-community__Qwen3.5-35B-A3B-4bit")
    try:
        models = manager.list_models()
    finally:
        runtime_state.clear_marker(marker)

    model = next(m for m in models if m.name == "mlx-community__Qwen3.5-35B-A3B-4bit")
    assert model.state.value == "running"
    assert model.loadable is True


def test_list_models_marks_unsupported_model_as_not_loadable(manager, tmp_models_dir, monkeypatch):
    """A complete model with an unsupported architecture should not appear loadable."""
    fake_model = tmp_models_dir / "downloaded" / "mlx-community__UnsupportedGemma4"
    fake_model.mkdir()
    (fake_model / "config.json").write_text('{"model_type":"gemma4"}')
    (fake_model / "tokenizer_config.json").write_text("{}")

    registry = {
        "mlx-community__UnsupportedGemma4": {
            "name": "mlx-community__UnsupportedGemma4",
            "repo_id": "mlx-community/UnsupportedGemma4",
            "path": str(fake_model),
            "source": "downloaded",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
        }
    }
    (tmp_models_dir / "registry.json").write_text(json.dumps(registry))

    monkeypatch.setattr("app.services.model_manager._supported_mlx_model_types", lambda: {"gemma", "gemma2", "gemma3"})
    monkeypatch.setattr("app.services.model_manager._mlx_model_remapping", lambda: {})
    monkeypatch.setattr("app.services.model_manager._supported_mlx_vlm_model_types", lambda: set())

    models = manager.list_models()
    model = next(m for m in models if m.name == "mlx-community__UnsupportedGemma4")

    assert model.state.value == "unsupported"
    assert model.loadable is False


def test_download_keyboard_interrupt_cleans_partial_dir_and_marker(manager, monkeypatch):
    """Interrupting a download should remove partial files and live state."""
    from app.core.exceptions import DownloadError

    repo_id = "mlx-community/InterruptedModel"
    local_name = "mlx-community__InterruptedModel"
    dest = manager._cfg.downloaded_models_path / local_name

    def fake_snapshot_download(**kwargs):
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")
        raise KeyboardInterrupt

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    with pytest.raises(DownloadError, match="interrupted by user"):
        manager.download(repo_id)

    assert not dest.exists()
    assert manager._runtime_state.snapshot() == {}


def test_ensure_model_loadable_rejects_unsupported_model(manager, tmp_models_dir, monkeypatch):
    """Loading an unsupported architecture should fail before mlx_lm.load()."""
    from app.core.exceptions import UnsupportedModelError

    fake_model = tmp_models_dir / "downloaded" / "mlx-community__UnsupportedGemma4"
    fake_model.mkdir()
    (fake_model / "config.json").write_text('{"model_type":"gemma4"}')
    (fake_model / "tokenizer_config.json").write_text("{}")

    registry = {
        "mlx-community__UnsupportedGemma4": {
            "name": "mlx-community__UnsupportedGemma4",
            "repo_id": "mlx-community/UnsupportedGemma4",
            "path": str(fake_model),
            "source": "downloaded",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
        }
    }
    (tmp_models_dir / "registry.json").write_text(json.dumps(registry))

    monkeypatch.setattr("app.services.model_manager._supported_mlx_model_types", lambda: {"gemma", "gemma2", "gemma3"})
    monkeypatch.setattr("app.services.model_manager._mlx_model_remapping", lambda: {})

    with pytest.raises(UnsupportedModelError):
        manager.ensure_model_loadable("mlx-community__UnsupportedGemma4")


def test_ensure_model_files_ready_allows_unsupported_mlx_lm_model(manager, tmp_models_dir, monkeypatch):
    """Multimodal-capable models should still pass the file-readiness check."""
    fake_model = tmp_models_dir / "downloaded" / "mlx-community__VisionGemma4"
    fake_model.mkdir()
    (fake_model / "config.json").write_text('{"model_type":"gemma4"}')
    (fake_model / "tokenizer_config.json").write_text("{}")

    registry = {
        "mlx-community__VisionGemma4": {
            "name": "mlx-community__VisionGemma4",
            "repo_id": "mlx-community/VisionGemma4",
            "path": str(fake_model),
            "source": "downloaded",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
        }
    }
    (tmp_models_dir / "registry.json").write_text(json.dumps(registry))

    monkeypatch.setattr("app.services.model_manager._supported_mlx_model_types", lambda: {"gemma", "gemma2", "gemma3"})
    monkeypatch.setattr("app.services.model_manager._mlx_model_remapping", lambda: {})

    info = manager.ensure_model_files_ready("mlx-community__VisionGemma4")
    assert info.name == "mlx-community__VisionGemma4"


def test_diagnose_model_reports_unsupported_runtime(manager, tmp_models_dir, monkeypatch):
    """Doctor should explain when mlx_lm lacks support for a model_type."""
    fake_model = tmp_models_dir / "downloaded" / "mlx-community__UnsupportedGemma4"
    fake_model.mkdir()
    (fake_model / "config.json").write_text('{"model_type":"gemma4"}')
    (fake_model / "tokenizer_config.json").write_text("{}")

    registry = {
        "mlx-community__UnsupportedGemma4": {
            "name": "mlx-community__UnsupportedGemma4",
            "repo_id": "mlx-community/UnsupportedGemma4",
            "path": str(fake_model),
            "source": "downloaded",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
        }
    }
    (tmp_models_dir / "registry.json").write_text(json.dumps(registry))

    monkeypatch.setattr("app.services.model_manager._supported_mlx_model_types", lambda: {"gemma", "gemma2", "gemma3"})
    monkeypatch.setattr("app.services.model_manager._mlx_model_remapping", lambda: {})
    monkeypatch.setattr("app.services.model_manager._supported_mlx_vlm_model_types", lambda: set())

    diagnosis = manager.diagnose_model("mlx-community__UnsupportedGemma4")

    assert diagnosis.model_type == "gemma4"
    assert diagnosis.supported_by_mlx is False
    assert "does not support" in diagnosis.summary
    assert diagnosis.recommendations


def test_diagnose_vlm_only_model_supported_via_mlx_vlm(manager, tmp_models_dir, monkeypatch):
    """A VLM-only architecture is validated against mlx-vlm, not mlx-lm.

    Loading routes purely on the detected backend, so a model mlx-lm does not know
    about but mlx-vlm does must still be reported as loadable rather than flagged
    unsupported with an irrelevant 'upgrade mlx-lm' recommendation.
    """
    fake_model = tmp_models_dir / "downloaded" / "mlx-community__VlmOnly"
    fake_model.mkdir()
    (fake_model / "config.json").write_text(
        json.dumps({"model_type": "llava", "vision_config": {"image_size": 336}})
    )
    (fake_model / "tokenizer_config.json").write_text("{}")

    registry = {
        "mlx-community__VlmOnly": {
            "name": "mlx-community__VlmOnly",
            "repo_id": "mlx-community/VlmOnly",
            "path": str(fake_model),
            "source": "downloaded",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
        }
    }
    (tmp_models_dir / "registry.json").write_text(json.dumps(registry))

    monkeypatch.setattr("app.services.model_manager._supported_mlx_model_types", lambda: {"gemma", "llama"})
    monkeypatch.setattr("app.services.model_manager._mlx_model_remapping", lambda: {})
    monkeypatch.setattr("app.services.model_manager._supported_mlx_vlm_model_types", lambda: {"llava", "qwen2_vl"})

    diagnosis = manager.diagnose_model("mlx-community__VlmOnly")

    assert diagnosis.backend == "mlx-vlm"
    assert diagnosis.supported_by_mlx is True
    assert diagnosis.loadable is True
    assert diagnosis.state.value == "ready"
    assert diagnosis.summary == "Model looks ready to load."


def test_get_model_not_found(manager):
    """get_model() should raise ModelNotFoundError for unknown models."""
    from app.core.exceptions import ModelNotFoundError

    with pytest.raises(ModelNotFoundError):
        manager.get_model("nonexistent-model")


def test_delete_model(manager, tmp_models_dir):
    """Delete should remove the directory and unregister the model."""
    fake_model = tmp_models_dir / "downloaded" / "mlx-community__DeleteMe"
    fake_model.mkdir()
    (fake_model / "config.json").write_text("{}")
    (fake_model / "tokenizer_config.json").write_text("{}")

    registry = {
        "mlx-community__DeleteMe": {
            "name": "mlx-community__DeleteMe",
            "repo_id": "mlx-community/DeleteMe",
            "path": str(fake_model),
            "source": "downloaded",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
        }
    }
    (tmp_models_dir / "registry.json").write_text(json.dumps(registry))

    manager.delete("mlx-community__DeleteMe")

    assert not fake_model.exists()
    assert not any(m.name == "mlx-community__DeleteMe" for m in manager.list_models())


def test_delete_custom_blocked_by_default(manager, tmp_models_dir):
    """Deleting a custom model without allow_custom should fail."""
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


def test_delete_running_model_blocked(manager, tmp_models_dir):
    """Deleting a running model should fail with a busy error."""
    from app.core.exceptions import ModelBusyError
    from app.services.model_runtime_state import ModelRuntimeState

    fake_model = tmp_models_dir / "downloaded" / "mlx-community__BusyDelete"
    fake_model.mkdir()
    (fake_model / "config.json").write_text("{}")
    (fake_model / "tokenizer_config.json").write_text("{}")

    registry = {
        "mlx-community__BusyDelete": {
            "name": "mlx-community__BusyDelete",
            "repo_id": "mlx-community/BusyDelete",
            "path": str(fake_model),
            "source": "downloaded",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
        }
    }
    (tmp_models_dir / "registry.json").write_text(json.dumps(registry))

    runtime_state = ModelRuntimeState(cfg=manager._cfg)
    marker = runtime_state.mark_running("mlx-community__BusyDelete")
    try:
        with pytest.raises(ModelBusyError):
            manager.delete("mlx-community__BusyDelete")
    finally:
        runtime_state.clear_marker(marker)


def test_update_running_model_blocked(manager, tmp_models_dir):
    """Updating a running model should fail with a busy error."""
    from app.core.exceptions import ModelBusyError
    from app.services.model_runtime_state import ModelRuntimeState

    fake_model = tmp_models_dir / "downloaded" / "mlx-community__BusyUpdate"
    fake_model.mkdir()
    (fake_model / "config.json").write_text("{}")
    (fake_model / "tokenizer_config.json").write_text("{}")

    registry = {
        "mlx-community__BusyUpdate": {
            "name": "mlx-community__BusyUpdate",
            "repo_id": "mlx-community/BusyUpdate",
            "path": str(fake_model),
            "source": "downloaded",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
        }
    }
    (tmp_models_dir / "registry.json").write_text(json.dumps(registry))

    runtime_state = ModelRuntimeState(cfg=manager._cfg)
    marker = runtime_state.mark_running("mlx-community__BusyUpdate")
    try:
        with pytest.raises(ModelBusyError):
            manager.update("mlx-community__BusyUpdate")
    finally:
        runtime_state.clear_marker(marker)


def test_update_keyboard_interrupt_preserves_existing_model(manager, tmp_models_dir, monkeypatch):
    """Interrupting an update should leave the existing model installed."""
    from app.core.exceptions import DownloadError

    fake_model = tmp_models_dir / "downloaded" / "mlx-community__InterruptedUpdate"
    fake_model.mkdir()
    (fake_model / "config.json").write_text('{"old": true}')
    (fake_model / "tokenizer_config.json").write_text("{}")

    registry = {
        "mlx-community__InterruptedUpdate": {
            "name": "mlx-community__InterruptedUpdate",
            "repo_id": "mlx-community/InterruptedUpdate",
            "path": str(fake_model),
            "source": "downloaded",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
        }
    }
    registry_path = tmp_models_dir / "registry.json"
    registry_path.write_text(json.dumps(registry))

    def fake_snapshot_download(**kwargs):
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text('{"new": true}')
        raise KeyboardInterrupt

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    with pytest.raises(DownloadError, match="interrupted by user"):
        manager.update("mlx-community__InterruptedUpdate")

    assert fake_model.exists()
    assert (fake_model / "config.json").read_text() == '{"old": true}'
    assert json.loads(registry_path.read_text()) == registry
    assert manager._runtime_state.snapshot() == {}
    assert not any((tmp_models_dir / "runtime" / "updates").iterdir())


def test_delete_downloading_model_blocked(manager, tmp_models_dir):
    """Deleting a downloading model should fail with a busy error."""
    from app.core.exceptions import ModelBusyError
    from app.services.model_runtime_state import ModelRuntimeState

    fake_model = tmp_models_dir / "downloaded" / "mlx-community__BusyDownloadingDelete"
    fake_model.mkdir()
    (fake_model / "config.json").write_text("{}")
    (fake_model / "tokenizer_config.json").write_text("{}")

    registry = {
        "mlx-community__BusyDownloadingDelete": {
            "name": "mlx-community__BusyDownloadingDelete",
            "repo_id": "mlx-community/BusyDownloadingDelete",
            "path": str(fake_model),
            "source": "downloaded",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
        }
    }
    (tmp_models_dir / "registry.json").write_text(json.dumps(registry))

    runtime_state = ModelRuntimeState(cfg=manager._cfg)
    marker = runtime_state.mark_downloading("mlx-community__BusyDownloadingDelete")
    try:
        with pytest.raises(ModelBusyError):
            manager.delete("mlx-community__BusyDownloadingDelete")
    finally:
        runtime_state.clear_marker(marker)


def test_path_traversal_blocked(manager):
    """_safe_downloaded_path() must reject traversal attempts."""
    from app.core.exceptions import InvalidModelPathError

    with pytest.raises(InvalidModelPathError):
        manager._safe_downloaded_path("../../etc/passwd")
