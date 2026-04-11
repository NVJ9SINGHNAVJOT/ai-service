"""
Focused tests for the mlx-vlm wrapper used by chat-media.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def test_media_chat_validate_requires_installed_mlx_vlm(tmp_path: Path, monkeypatch):
    """A friendly error should be raised when mlx-vlm is missing."""
    from app.core.exceptions import MediaChatError
    from app.services.media_chat_session import MediaChatSession

    model_dir = tmp_path / "model"
    image_path = tmp_path / "image.jpg"
    model_dir.mkdir()
    image_path.write_bytes(b"fake")

    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

    session = MediaChatSession(
        model_path=model_dir,
        model_name="media-model",
        image_path=image_path,
        allowed_modalities=["image"],
    )

    with pytest.raises(MediaChatError):
        session.run()


def test_media_chat_set_image_updates_current_image(tmp_path: Path, monkeypatch):
    """Setting an image should validate and remember the active image path."""
    from app.services.media_chat_session import MediaChatSession

    model_dir = tmp_path / "model"
    image_path = tmp_path / "image.jpg"
    model_dir.mkdir()
    image_path.write_bytes(b"fake")

    session = MediaChatSession(
        model_path=model_dir,
        model_name="media-model",
        allowed_modalities=["image"],
    )
    session._load_image = lambda path: object()

    session._set_image(image_path)

    assert session._current_image_path == str(image_path)


def test_media_chat_generate_response_streams_chunks(tmp_path: Path):
    """Responses should be built incrementally from streamed mlx-vlm chunks."""
    from app.services.media_chat_session import MediaChatSession

    model_dir = tmp_path / "model"
    image_path = tmp_path / "image.jpg"
    model_dir.mkdir()
    image_path.write_bytes(b"fake")

    session = MediaChatSession(
        model_path=model_dir,
        model_name="media-model",
        image_path=image_path,
        allowed_modalities=["image"],
    )
    session._processor = object()
    session._model = SimpleNamespace(config=object())
    session._current_image_path = str(image_path)
    session._history = [session._message("user", "Describe the image")]
    session._vision_cache = object()
    session._prompt_cache_state = object()
    session._apply_chat_template = lambda processor, config, history, num_images, num_audios=0: "formatted prompt"
    session._stream_generate = lambda *args, **kwargs: iter(
        [
            SimpleNamespace(text="Hello ", finish_reason=None),
            SimpleNamespace(
                text="world",
                finish_reason="stop",
                prompt_tokens=5,
                generation_tokens=2,
                prompt_tps=10.0,
                generation_tps=5.0,
            ),
        ]
    )

    response, usage = session._generate_response()

    assert response == "Hello world"
    assert usage is not None
    assert usage["completion_tokens"] == 2


def test_media_chat_handle_clear_preserves_system_prompt(tmp_path: Path):
    """Clearing should reset history but keep the configured system prompt."""
    from app.services.media_chat_session import MediaChatSession

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    class PromptCacheState:
        pass

    session = MediaChatSession(
        model_path=model_dir,
        model_name="media-model",
        system_prompt="You are a helpful media assistant.",
        allowed_modalities=["image"],
    )
    session._prompt_cache_state = PromptCacheState()
    session._history = [
        session._message("system", "You are a helpful media assistant."),
        session._message("user", "hello"),
    ]

    keep_running = session._handle_command("/clear")

    assert keep_running is True
    assert session._history == [session._message("system", "You are a helpful media assistant.")]
    assert isinstance(session._prompt_cache_state, PromptCacheState)


def test_media_chat_run_loads_model_and_starts_loop(tmp_path: Path, monkeypatch):
    """The wrapper should load runtime pieces, preload the image, and start looping."""
    from app.services.model_runtime_state import ModelRuntimeState
    from app.services.media_chat_session import MediaChatSession
    from app.config import Settings

    model_dir = tmp_path / "model"
    image_path = tmp_path / "image.jpg"
    runtime_dir = tmp_path / "runtime"
    model_dir.mkdir()
    image_path.write_bytes(b"fake")

    calls: list[tuple[str, str | None]] = []

    def fake_import(name: str):
        if name == "mlx_vlm":
            return SimpleNamespace(load=lambda model_path: (SimpleNamespace(config=object()), object()))
        if name == "mlx_vlm.generate":
            return SimpleNamespace(
                stream_generate=lambda *args, **kwargs: iter([]),
                PromptCacheState=type("PromptCacheState", (), {}),
            )
        if name == "mlx_vlm.prompt_utils":
            return SimpleNamespace(apply_chat_template=lambda processor, config, history, num_images, num_audios=0: "prompt")
        if name == "mlx_vlm.utils":
            return SimpleNamespace(load_image=lambda path: calls.append(("image", path)) or object())
        if name == "mlx_vlm.vision_cache":
            return SimpleNamespace(VisionFeatureCache=lambda: object())
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr("importlib.import_module", fake_import)

    session = MediaChatSession(
        model_path=model_dir,
        model_name="media-model",
        image_path=image_path,
        system_prompt="system text",
        allowed_modalities=["image"],
    )
    session._runtime_state = ModelRuntimeState(
        cfg=Settings(
            downloaded_models_dir=str(tmp_path / "downloaded"),
            custom_models_dir=str(tmp_path / "custom"),
            model_registry_file=str(tmp_path / "registry.json"),
            model_runtime_dir=str(runtime_dir),
        )
    )

    monkeypatch.setattr(
        session,
        "_loop",
        lambda: calls.append(("loop", None)),
    )

    session.run()

    assert ("image", str(image_path)) in calls
    assert ("loop", None) in calls
    assert session._history[0] == session._message("system", "system text")


def test_media_chat_set_audio_updates_current_audio(tmp_path: Path):
    """Setting audio should validate and remember the active audio path."""
    from app.services.media_chat_session import MediaChatSession

    model_dir = tmp_path / "model"
    audio_path = tmp_path / "clip.wav"
    model_dir.mkdir()
    audio_path.write_bytes(b"fake")

    session = MediaChatSession(
        model_path=model_dir,
        model_name="media-model",
        allowed_modalities=["audio"],
    )

    session._set_audio(audio_path)

    assert session._current_audio_path == str(audio_path)


def test_media_chat_rejects_unsupported_media_type(tmp_path: Path):
    """The session should fail fast when a model does not advertise a media type."""
    from app.core.exceptions import MediaChatError
    from app.services.media_chat_session import MediaChatSession

    model_dir = tmp_path / "model"
    image_path = tmp_path / "image.jpg"
    model_dir.mkdir()
    image_path.write_bytes(b"fake")

    session = MediaChatSession(
        model_path=model_dir,
        model_name="media-model",
        allowed_modalities=["audio"],
    )

    with pytest.raises(MediaChatError):
        session._set_image(image_path)


def test_media_chat_verbose_prints_stats(tmp_path: Path, monkeypatch):
    """Verbose mode should print token and timing stats after a reply."""
    from app.services.media_chat_session import MediaChatSession

    model_dir = tmp_path / "model"
    image_path = tmp_path / "image.jpg"
    model_dir.mkdir()
    image_path.write_bytes(b"fake")

    prompts = iter(["Describe", "quit"])
    printed: list[str] = []

    monkeypatch.setattr("app.services.media_chat_session.Prompt.ask", lambda _: next(prompts))
    monkeypatch.setattr(
        "app.services.media_chat_session.console.print",
        lambda *args, **kwargs: printed.append("" if not args else str(args[0])),
    )

    session = MediaChatSession(
        model_path=model_dir,
        model_name="media-model",
        image_path=image_path,
        allowed_modalities=["image"],
        verbose=True,
    )
    session._processor = object()
    session._model = SimpleNamespace(config=object())
    session._current_image_path = str(image_path)
    session._history = []
    session._apply_chat_template = lambda processor, config, history, num_images, num_audios=0: "formatted prompt"
    session._stream_generate = lambda *args, **kwargs: iter(
        [
            SimpleNamespace(text="Hello ", finish_reason=None),
            SimpleNamespace(
                text="world",
                finish_reason="stop",
                prompt_tokens=4,
                generation_tokens=2,
                prompt_tps=12.5,
                generation_tps=6.25,
            ),
        ]
    )
    session._last_load_duration_s = 0.25

    session._loop()

    assert any("prompt eval count:" in line for line in printed)
    assert any("eval rate:" in line for line in printed)


def test_media_chat_generate_response_builds_best_effort_usage_without_finish_reason(tmp_path: Path):
    """Verbose stats should still have timing data even if mlx-vlm omits a final finish marker."""
    from app.services.media_chat_session import MediaChatSession

    model_dir = tmp_path / "model"
    image_path = tmp_path / "image.jpg"
    model_dir.mkdir()
    image_path.write_bytes(b"fake")

    session = MediaChatSession(
        model_path=model_dir,
        model_name="media-model",
        image_path=image_path,
        allowed_modalities=["image"],
    )
    session._processor = object()
    session._model = SimpleNamespace(config=object())
    session._current_image_path = str(image_path)
    session._history = [session._message("user", "Describe the image")]
    session._vision_cache = object()
    session._prompt_cache_state = object()
    session._apply_chat_template = lambda processor, config, history, num_images, num_audios=0: "formatted prompt"
    session._stream_generate = lambda *args, **kwargs: iter(
        [
            SimpleNamespace(text="Hello ", prompt_tokens=5, generation_tokens=1, prompt_tps=9.0, generation_tps=4.5),
            SimpleNamespace(text="world", prompt_tokens=5, generation_tokens=2, prompt_tps=9.0, generation_tps=4.5),
        ]
    )

    response, usage = session._generate_response()

    assert response == "Hello world"
    assert usage is not None
    assert usage["prompt_tokens"] == 5
    assert usage["completion_tokens"] == 2
    assert usage["metrics"]["total_duration_s"] is not None
