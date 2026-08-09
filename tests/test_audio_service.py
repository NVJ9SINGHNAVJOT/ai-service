"""
Tests for AudioService internals — STT model selection across the two backends,
the STT/TTS idle-unload lifecycle, and the local cache gate that keeps a request
from ever triggering a model download.

These drive the service directly with fake mlx-whisper / mlx-audio models (no
download), and verify that a resident handle is dropped after its idle timeout
and reloaded lazily on the next request.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.config import Settings
from app.core.exceptions import (
    InvalidLangCodeError,
    InvalidSTTModelError,
    InvalidVoiceError,
    SpeechModelNotPreparedError,
)
from app.services.audio import AudioService

WHISPER = "mlx-community/whisper-large-v3-turbo"
PARAKEET = "mlx-community/parakeet-tdt-0.6b-v2"


def _install_fake_mlx_audio(monkeypatch) -> dict:
    """Fake mlx_audio.utils.load_model (TTS); return a dict tracking load count."""
    load_calls = {"n": 0}

    class _FakeModel:
        def generate(self, **kwargs):
            load_calls["last_generate"] = kwargs
            yield SimpleNamespace(audio=np.zeros(4, dtype=np.float32), sample_rate=24000)

    def _load_model(_name):
        load_calls["n"] += 1
        return _FakeModel()

    fake_utils = SimpleNamespace(load_model=_load_model)
    monkeypatch.setitem(sys.modules, "mlx_audio", SimpleNamespace(utils=fake_utils))
    monkeypatch.setitem(sys.modules, "mlx_audio.utils", fake_utils)
    # espeak + upstream patch are best-effort side-shows; neutralize them.
    monkeypatch.setattr(AudioService, "_configure_espeak", staticmethod(lambda: None))
    monkeypatch.setattr("app.services.audio.patch_interpolate_ceil_drift", lambda: None)
    return load_calls


def _install_fake_parakeet(monkeypatch, text: str = "  hi  ") -> dict:
    """Fake mlx_audio.stt.utils.load_model; return a dict tracking calls."""
    calls: dict = {"n": 0}

    class _FakeModel:
        def generate(self, audio, **kwargs):
            calls["generate"] = {"audio": audio, **kwargs}
            return SimpleNamespace(text=text)

    def _load_model(model_path, **kwargs):
        calls["n"] += 1
        calls["load"] = {"model_path": model_path, **kwargs}
        return _FakeModel()

    fake_utils = SimpleNamespace(load_model=_load_model)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", fake_utils)
    return calls


def _install_fake_whisper(monkeypatch, text: str = "  hi  ") -> dict:
    """Fake mlx_whisper plus the ModelHolder handle cache the STT slot governs."""
    calls: dict = {"n": 0}

    class _ModelHolder:
        model = None
        model_path = None

        @classmethod
        def get_model(cls, model_path, dtype):
            calls["n"] += 1
            cls.model, cls.model_path = object(), model_path
            return cls.model

    def _transcribe(audio_path, path_or_hf_repo=None, language=None):
        calls["transcribe"] = {"audio_path": audio_path, "path": path_or_hf_repo, "language": language}
        return {"text": text}

    monkeypatch.setitem(
        sys.modules, "mlx_whisper", SimpleNamespace(transcribe=_transcribe)
    )
    monkeypatch.setitem(
        sys.modules, "mlx_whisper.transcribe", SimpleNamespace(ModelHolder=_ModelHolder)
    )
    calls["holder"] = _ModelHolder
    return calls


def _install_fake_snapshot(monkeypatch, tmp_path: Path, voices=("af_heart",)) -> Path:
    """Stand in for the local HF cache probe with a tmp snapshot dir."""
    snapshot = tmp_path / "snapshot"
    (snapshot / "voices").mkdir(parents=True)
    for voice in voices:
        (snapshot / "voices" / f"{voice}.safetensors").touch()
    monkeypatch.setattr(
        AudioService, "_ensure_speech_model_available", lambda self, repo, label: snapshot
    )
    return snapshot


def _install_fake_tts(monkeypatch, tmp_path: Path, voices=("af_heart",)) -> dict:
    """A fully faked TTS stack: cached snapshot + mlx-audio model."""
    _install_fake_snapshot(monkeypatch, tmp_path, voices)
    return _install_fake_mlx_audio(monkeypatch)


def _install_empty_hf_cache(monkeypatch) -> None:
    """Fake huggingface_hub so the cache probe finds nothing locally."""

    def _snapshot_download(**_kwargs):
        raise OSError("Cannot find an appropriate cached snapshot folder.")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=_snapshot_download)
    )


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    """Poll until predicate() is true or the timeout elapses."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_tts_unloads_after_idle_timeout_and_reloads(monkeypatch, tmp_path):
    """The TTS model drops after the idle timeout and reloads on the next call."""
    load_calls = _install_fake_tts(monkeypatch, tmp_path)
    svc = AudioService(Settings(tts_idle_timeout_seconds=0.2))

    svc.synthesize("hello")
    assert svc._tts.loaded_name is not None
    assert load_calls["n"] == 1

    assert _wait_until(lambda: svc._tts.loaded_name is None), "TTS should unload after idle timeout"

    svc.synthesize("again")
    assert svc._tts.loaded_name is not None
    assert load_calls["n"] == 2  # reloaded lazily on the next request


def test_tts_stays_resident_when_timeout_disabled(monkeypatch, tmp_path):
    """A timeout of 0 keeps the TTS model resident and arms no timer."""
    load_calls = _install_fake_tts(monkeypatch, tmp_path)
    svc = AudioService(Settings(tts_idle_timeout_seconds=0))

    svc.synthesize("hello")

    assert svc._tts._timer is None  # disabled → no unload scheduled
    time.sleep(0.1)
    assert svc._tts.loaded_name is not None
    assert load_calls["n"] == 1


def test_stt_unloads_after_idle_timeout_and_reloads(monkeypatch, tmp_path):
    """The STT model drops after its own idle timeout and reloads on the next call."""
    _install_fake_snapshot(monkeypatch, tmp_path)
    whisper = _install_fake_whisper(monkeypatch)
    svc = AudioService(Settings(stt_idle_timeout_seconds=0.2))

    svc.transcribe(tmp_path / "clip.wav")
    assert svc._stt.loaded_name == WHISPER
    assert whisper["n"] == 1

    assert _wait_until(lambda: svc._stt.loaded_name is None), "STT should unload after idle timeout"
    # the slot also clears mlx-whisper's own module-level handle
    assert whisper["holder"].model is None

    svc.transcribe(tmp_path / "clip.wav")
    assert whisper["n"] == 2


def test_stt_stays_resident_when_timeout_disabled(monkeypatch, tmp_path):
    """A timeout of 0 keeps the STT model resident and arms no timer."""
    _install_fake_snapshot(monkeypatch, tmp_path)
    whisper = _install_fake_whisper(monkeypatch)
    svc = AudioService(Settings(stt_idle_timeout_seconds=0))

    svc.transcribe(tmp_path / "clip.wav")

    assert svc._stt._timer is None
    time.sleep(0.1)
    assert svc._stt.loaded_name == WHISPER
    assert whisper["n"] == 1


# ── per-request STT model selection ──────────────────────────────────────────


def test_transcribe_routes_parakeet_repo_to_mlx_audio(monkeypatch, tmp_path):
    """A parakeet repo loads through mlx-audio, by path and with an explicit model_type."""
    snapshot = _install_fake_snapshot(monkeypatch, tmp_path)
    parakeet = _install_fake_parakeet(monkeypatch)
    svc = AudioService(Settings(stt_idle_timeout_seconds=0))

    assert svc.transcribe(tmp_path / "clip.wav", model=PARAKEET) == "hi"

    # by resolved path, so mlx-audio can't fall back to downloading …
    assert parakeet["load"]["model_path"] == str(snapshot)
    # … and named explicitly, since parakeet's NeMo config.json has no model_type
    assert parakeet["load"]["model_type"] == "parakeet"
    assert parakeet["generate"]["chunk_duration"] > 0


def test_transcribe_ignores_language_hint_for_parakeet(monkeypatch, tmp_path):
    """Parakeet takes no language argument, so the hint is dropped rather than passed."""
    _install_fake_snapshot(monkeypatch, tmp_path)
    parakeet = _install_fake_parakeet(monkeypatch)
    svc = AudioService(Settings(stt_idle_timeout_seconds=0))

    svc.transcribe(tmp_path / "clip.wav", language="en", model=PARAKEET)

    assert "language" not in parakeet["generate"]


def test_transcribe_rejects_unconfigured_model(monkeypatch, tmp_path):
    """An unknown model is a bad name (400), not a missing download (503)."""
    _install_fake_snapshot(monkeypatch, tmp_path)
    _install_fake_whisper(monkeypatch)
    svc = AudioService(Settings())

    with pytest.raises(InvalidSTTModelError) as excinfo:
        svc.transcribe(tmp_path / "clip.wav", model="mlx-community/not-configured")

    message = str(excinfo.value)
    assert "not-configured" in message
    assert WHISPER in message  # the caller is told what it can use
    assert "audio:setup" not in message  # running setup would not help


def test_transcribe_rejects_sanitized_model_name(monkeypatch, tmp_path):
    """The audio API takes HF repo ids; `org__name` is on-disk naming for models/."""
    _install_fake_snapshot(monkeypatch, tmp_path)
    _install_fake_whisper(monkeypatch)
    svc = AudioService(Settings())

    with pytest.raises(InvalidSTTModelError):
        svc.transcribe(tmp_path / "clip.wav", model=WHISPER.replace("/", "__"))


def test_transcribe_switching_model_unloads_the_previous_one(monkeypatch, tmp_path):
    """Only one STT model is resident: switching drops the other."""
    _install_fake_snapshot(monkeypatch, tmp_path)
    whisper = _install_fake_whisper(monkeypatch)
    parakeet = _install_fake_parakeet(monkeypatch)
    svc = AudioService(Settings(stt_idle_timeout_seconds=0))

    svc.transcribe(tmp_path / "clip.wav", model=WHISPER)
    assert svc._stt.loaded_name == WHISPER

    svc.transcribe(tmp_path / "clip.wav", model=PARAKEET)
    assert svc._stt.loaded_name == PARAKEET
    assert whisper["holder"].model is None  # whisper's cache was cleared on swap

    svc.transcribe(tmp_path / "clip.wav", model=WHISPER)
    assert whisper["n"] == 2  # reloaded from scratch
    assert parakeet["n"] == 1


# ── the local-cache gate: a request must never download ──────────────────────


def test_transcribe_raises_when_stt_model_not_cached(monkeypatch, tmp_path):
    """A missing Whisper snapshot fails fast instead of downloading mid-request."""
    _install_empty_hf_cache(monkeypatch)
    whisper = _install_fake_whisper(monkeypatch)
    svc = AudioService(Settings(stt_model="mlx-community/whisper-not-here"))

    with pytest.raises(SpeechModelNotPreparedError) as excinfo:
        svc.transcribe(tmp_path / "clip.wav")

    assert "whisper-not-here" in str(excinfo.value)
    assert "task audio:setup" in str(excinfo.value)
    assert whisper["n"] == 0  # never reached the backend


def test_transcribe_passes_resolved_snapshot_path(monkeypatch, tmp_path):
    """Whisper gets the local snapshot dir, not the repo id — so it cannot download."""
    snapshot = _install_fake_snapshot(monkeypatch, tmp_path)
    whisper = _install_fake_whisper(monkeypatch)

    assert AudioService(Settings()).transcribe(tmp_path / "clip.wav") == "hi"
    assert whisper["transcribe"]["path"] == str(snapshot)


def test_synthesize_raises_when_tts_model_not_cached(monkeypatch, tmp_path):
    """A missing Kokoro snapshot fails fast; mlx-audio is never asked to load."""
    load_calls = _install_fake_mlx_audio(monkeypatch)
    _install_empty_hf_cache(monkeypatch)
    svc = AudioService(Settings(tts_model="prince-canuma/Kokoro-not-here"))

    with pytest.raises(SpeechModelNotPreparedError) as excinfo:
        svc.synthesize("hello")

    assert "Kokoro-not-here" in str(excinfo.value)
    assert load_calls["n"] == 0


def test_synthesize_raises_when_no_voice_packs_cached(monkeypatch, tmp_path):
    """Kokoro fetches voice packs lazily, so an empty voices/ dir is a setup problem."""
    _install_fake_tts(monkeypatch, tmp_path, voices=())
    svc = AudioService(Settings(tts_idle_timeout_seconds=0))

    with pytest.raises(SpeechModelNotPreparedError) as excinfo:
        svc.synthesize("hello", voice="af_bella")

    assert "af_bella" in str(excinfo.value)


def test_synthesize_rejects_unknown_voice_when_cache_is_prepared(monkeypatch, tmp_path):
    """With voice packs present, an absent voice is a bad name — not a missing download."""
    _install_fake_tts(monkeypatch, tmp_path, voices=("af_heart", "af_bella"))
    svc = AudioService(Settings(tts_idle_timeout_seconds=0))

    with pytest.raises(InvalidVoiceError) as excinfo:
        svc.synthesize("hello", voice="zz_bogus")

    message = str(excinfo.value)
    assert "zz_bogus" in message
    assert "af_bella, af_heart" in message  # the caller is told what it can use
    assert "audio:setup" not in message  # running setup would not help


# ── per-request TTS language code ────────────────────────────────────────────


def test_synthesize_passes_request_lang_code(monkeypatch, tmp_path):
    """A request-level lang_code overrides the configured default."""
    load_calls = _install_fake_tts(monkeypatch, tmp_path)
    svc = AudioService(Settings(tts_lang_code="a", tts_idle_timeout_seconds=0))

    svc.synthesize("hello", lang_code="b")

    assert load_calls["last_generate"]["lang_code"] == "b"


def test_synthesize_rejects_unknown_lang_code(monkeypatch, tmp_path):
    """An unsupported code is a bad name (400), caught before the model is loaded."""
    load_calls = _install_fake_tts(monkeypatch, tmp_path)
    svc = AudioService(Settings(tts_idle_timeout_seconds=0))

    with pytest.raises(InvalidLangCodeError) as excinfo:
        svc.synthesize("hello", lang_code="zz")

    assert "zz" in str(excinfo.value)
    assert load_calls["n"] == 0


# ── capability description (drives GET /v1/audio/models) ─────────────────────


def test_describe_reports_readiness_without_loading(monkeypatch, tmp_path):
    """An uncached model is reported as not-ready rather than raising."""
    _install_empty_hf_cache(monkeypatch)
    load_calls = _install_fake_mlx_audio(monkeypatch)
    svc = AudioService(Settings())

    stt = svc.describe_stt()
    tts = svc.describe_tts()

    assert stt["default"] == WHISPER
    assert [m["id"] for m in stt["models"]] == svc._cfg.available_stt_models
    assert all(m["ready"] is False and m["loaded"] is False for m in stt["models"])
    assert tts["ready"] is False and tts["voices"] == []
    assert load_calls["n"] == 0  # describing never loads


def test_describe_stt_reports_backend_and_language_support(monkeypatch, tmp_path):
    """Each model advertises the loader it needs and whether a language hint applies."""
    _install_fake_snapshot(monkeypatch, tmp_path)
    svc = AudioService(Settings())

    by_id = {m["id"]: m for m in svc.describe_stt()["models"]}

    assert by_id[WHISPER]["backend"] == "mlx-whisper"
    assert by_id[WHISPER]["accepts_language_hint"] is True
    assert by_id[PARAKEET]["backend"] == "mlx-audio"
    assert by_id[PARAKEET]["accepts_language_hint"] is False
    assert by_id[PARAKEET]["languages"] == ["en"]  # v2 is English-only


def test_describe_tts_lists_cached_voices_and_lang_codes(monkeypatch, tmp_path):
    """The frontend gets the voices actually present plus every supported code."""
    _install_fake_snapshot(monkeypatch, tmp_path, voices=("af_heart", "bf_emma"))
    svc = AudioService(Settings())

    tts = svc.describe_tts()

    assert tts["ready"] is True
    assert tts["voices"] == ["af_heart", "bf_emma"]
    assert {c["code"] for c in tts["lang_codes"]} >= {"a", "b"}
    assert tts["speed"] == {"min": 0.25, "max": 4.0, "default": 1.0}
