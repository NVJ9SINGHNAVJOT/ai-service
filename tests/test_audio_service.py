"""
Tests for AudioService internals — the TTS idle-unload lifecycle, and the local
cache gate that keeps a request from ever triggering a model download.

These drive the service directly with a fake mlx-audio model (no download), and
verify that the resident Kokoro handle is dropped after `tts_idle_timeout_seconds`
of inactivity and reloaded lazily on the next request.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.config import Settings
from app.core.exceptions import InvalidVoiceError, SpeechModelNotPreparedError
from app.services.audio import AudioService


def _install_fake_mlx_audio(monkeypatch) -> dict:
    """Fake mlx_audio.utils.load_model; return a dict tracking load count."""
    load_calls = {"n": 0}

    class _FakeModel:
        def generate(self, **_kwargs):
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
    assert svc._tts_model is not None
    assert load_calls["n"] == 1

    assert _wait_until(lambda: svc._tts_model is None), "TTS should unload after idle timeout"
    assert svc._tts_model_name is None

    svc.synthesize("again")
    assert svc._tts_model is not None
    assert load_calls["n"] == 2  # reloaded lazily on the next request


def test_tts_stays_resident_when_timeout_disabled(monkeypatch, tmp_path):
    """A timeout of 0 keeps the TTS model resident and arms no timer."""
    load_calls = _install_fake_tts(monkeypatch, tmp_path)
    svc = AudioService(Settings(tts_idle_timeout_seconds=0))

    svc.synthesize("hello")

    assert svc._tts_idle_timer is None  # disabled → no unload scheduled
    time.sleep(0.1)
    assert svc._tts_model is not None
    assert load_calls["n"] == 1


# ── the local-cache gate: a request must never download ──────────────────────


def test_transcribe_raises_when_stt_model_not_cached(monkeypatch, tmp_path):
    """A missing Whisper snapshot fails fast instead of downloading mid-request."""
    _install_empty_hf_cache(monkeypatch)
    transcribe_calls = {"n": 0}
    monkeypatch.setitem(
        sys.modules,
        "mlx_whisper",
        SimpleNamespace(transcribe=lambda *a, **k: transcribe_calls.__setitem__("n", 1)),
    )
    svc = AudioService(Settings(stt_model="mlx-community/whisper-not-here"))

    with pytest.raises(SpeechModelNotPreparedError) as excinfo:
        svc.transcribe(tmp_path / "clip.wav")

    assert "whisper-not-here" in str(excinfo.value)
    assert "task audio:setup" in str(excinfo.value)
    assert transcribe_calls["n"] == 0  # never reached the backend


def test_transcribe_passes_resolved_snapshot_path(monkeypatch, tmp_path):
    """Whisper gets the local snapshot dir, not the repo id — so it cannot download."""
    snapshot = _install_fake_snapshot(monkeypatch, tmp_path)
    seen = {}

    def _transcribe(audio_path, path_or_hf_repo=None, language=None):
        seen["path_or_hf_repo"] = path_or_hf_repo
        return {"text": "  hi  "}

    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=_transcribe))

    assert AudioService(Settings()).transcribe(tmp_path / "clip.wav") == "hi"
    assert seen["path_or_hf_repo"] == str(snapshot)


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
