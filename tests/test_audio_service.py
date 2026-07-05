"""
Tests for AudioService internals — specifically the TTS idle-unload lifecycle.

These drive the service directly with a fake mlx-audio model (no download), and
verify that the resident Kokoro handle is dropped after `tts_idle_timeout_seconds`
of inactivity and reloaded lazily on the next request.
"""

from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import numpy as np

from app.config import Settings
from app.services.audio import AudioService


def _install_fake_tts(monkeypatch) -> dict:
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


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    """Poll until predicate() is true or the timeout elapses."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_tts_unloads_after_idle_timeout_and_reloads(monkeypatch):
    """The TTS model drops after the idle timeout and reloads on the next call."""
    load_calls = _install_fake_tts(monkeypatch)
    svc = AudioService(Settings(tts_idle_timeout_seconds=0.2))

    svc.synthesize("hello")
    assert svc._tts_model is not None
    assert load_calls["n"] == 1

    assert _wait_until(lambda: svc._tts_model is None), "TTS should unload after idle timeout"
    assert svc._tts_model_name is None

    svc.synthesize("again")
    assert svc._tts_model is not None
    assert load_calls["n"] == 2  # reloaded lazily on the next request


def test_tts_stays_resident_when_timeout_disabled(monkeypatch):
    """A timeout of 0 keeps the TTS model resident and arms no timer."""
    load_calls = _install_fake_tts(monkeypatch)
    svc = AudioService(Settings(tts_idle_timeout_seconds=0))

    svc.synthesize("hello")

    assert svc._tts_idle_timer is None  # disabled → no unload scheduled
    time.sleep(0.1)
    assert svc._tts_model is not None
    assert load_calls["n"] == 1
