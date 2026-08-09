"""
Tests for the local audio (STT / TTS) endpoints.

These mock AudioService so they run without downloading Whisper/Kokoro models;
they exercise the request/response wiring, the OpenAI-compatible shapes, and —
via the multipart transcription test — guard against the logging middleware
re-consuming the request stream.
"""

from __future__ import annotations

import io
import logging

import numpy as np

from app.core.exceptions import InferenceError, InvalidVoiceError, SpeechModelNotPreparedError


def test_speech_endpoint_returns_wav(api_client, monkeypatch):
    """POST /v1/audio/speech returns WAV audio bytes."""
    import app.main as main

    monkeypatch.setattr(
        main.audio_service,
        "synthesize",
        lambda text, voice=None, speed=1.0: (np.zeros(2400, dtype=np.float32), 24000),
    )

    resp = api_client.post("/v1/audio/speech", json={"input": "Hello there.", "voice": "af_heart"})

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.content[:4] == b"RIFF"  # valid WAV container


def test_speech_endpoint_rejects_non_wav_format(api_client):
    """Only response_format='wav' is supported; others return 400."""
    resp = api_client.post("/v1/audio/speech", json={"input": "hi", "response_format": "mp3"})

    assert resp.status_code == 400


def test_speech_endpoint_propagates_inference_error(api_client, monkeypatch):
    """A synthesis failure surfaces as HTTP 500 with the error detail."""
    import app.main as main

    def _boom(text, voice=None, speed=1.0):
        raise InferenceError("input text is empty.")

    monkeypatch.setattr(main.audio_service, "synthesize", _boom)

    resp = api_client.post("/v1/audio/speech", json={"input": "   "})

    assert resp.status_code == 500
    assert "input text is empty" in resp.json()["detail"]


def test_transcription_endpoint_returns_text(api_client, monkeypatch):
    """POST /v1/audio/transcriptions returns {'text': ...} for an uploaded clip.

    This also verifies the route still receives the multipart body — i.e. the
    logging middleware does not consume the request stream.
    """
    import app.main as main

    monkeypatch.setattr(
        main.audio_service,
        "transcribe",
        lambda audio_path, language=None: "transcribed text",
    )

    files = {"file": ("clip.wav", io.BytesIO(b"RIFFfake-wav-bytes"), "audio/wav")}
    resp = api_client.post("/v1/audio/transcriptions", files=files)

    assert resp.status_code == 200
    assert resp.json() == {"text": "transcribed text"}


def test_speech_endpoint_returns_503_when_model_not_prepared(api_client, monkeypatch, caplog):
    """An unprepared TTS model is a 503 pointing at `task audio:setup`, logged once."""
    import app.main as main

    def _not_prepared(text, voice=None, speed=1.0):
        raise SpeechModelNotPreparedError("TTS (Kokoro)", "prince-canuma/Kokoro-82M")

    monkeypatch.setattr(main.audio_service, "synthesize", _not_prepared)

    with caplog.at_level(logging.INFO):
        resp = api_client.post("/v1/audio/speech", json={"input": "hi"})

    assert resp.status_code == 503
    assert "task audio:setup" in resp.json()["detail"]

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1  # logged once at the boundary, not also in the route
    assert errors[0].request_id == resp.headers["X-Request-ID"]
    assert errors[0].exc_info is not None


def test_speech_endpoint_returns_400_for_unknown_voice(api_client, monkeypatch):
    """An unknown voice is the client's error (400), not an unprepared server (503)."""
    import app.main as main

    def _bad_voice(text, voice=None, speed=1.0):
        raise InvalidVoiceError("zz_bogus", ["af_bella", "af_heart"])

    monkeypatch.setattr(main.audio_service, "synthesize", _bad_voice)

    resp = api_client.post("/v1/audio/speech", json={"input": "hi", "voice": "zz_bogus"})

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "zz_bogus" in detail and "af_heart" in detail
    assert "audio:setup" not in detail  # not a setup problem, so don't suggest setup


def test_transcription_endpoint_returns_503_when_model_not_prepared(api_client, monkeypatch):
    """An unprepared STT model is a 503 rather than a mid-request download."""
    import app.main as main

    def _not_prepared(audio_path, language=None):
        raise SpeechModelNotPreparedError("STT (Whisper)", "mlx-community/whisper-large-v3-turbo")

    monkeypatch.setattr(main.audio_service, "transcribe", _not_prepared)

    files = {"file": ("clip.wav", io.BytesIO(b"RIFFfake-wav-bytes"), "audio/wav")}
    resp = api_client.post("/v1/audio/transcriptions", files=files)

    assert resp.status_code == 503
    assert "task audio:setup" in resp.json()["detail"]


def test_transcription_endpoint_requires_file(api_client):
    """Omitting the file field is a 422 validation error."""
    resp = api_client.post("/v1/audio/transcriptions")

    assert resp.status_code == 422


def test_speech_500_logged_once_with_request_id_and_traceback(api_client, monkeypatch, caplog):
    """A synthesis failure (500) is logged exactly once, correlated by request_id, with a traceback."""
    import app.main as main

    def _boom(text, voice=None, speed=1.0):
        raise InferenceError("input text is empty.")

    monkeypatch.setattr(main.audio_service, "synthesize", _boom)

    with caplog.at_level(logging.INFO):
        resp = api_client.post("/v1/audio/speech", json={"input": "   "})

    assert resp.status_code == 500

    request_id = resp.headers["X-Request-ID"]
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1  # logged once at the boundary, not also in the route
    record = errors[0]
    assert record.request_id == request_id  # id is stamped on the record (drives the log prefix)
    assert record.exc_info is not None  # 5xx carries the traceback
    # the diagnosis-era per-route log line is gone
    assert not any("Speech synthesis failed" in r.getMessage() for r in caplog.records)


def test_speech_400_logged_once_as_warning_with_traceback(api_client, caplog):
    """A 4xx HTTPException is logged once at warning level with the request_id and a traceback."""
    with caplog.at_level(logging.INFO):
        resp = api_client.post("/v1/audio/speech", json={"input": "hi", "response_format": "mp3"})

    assert resp.status_code == 400

    request_id = resp.headers["X-Request-ID"]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    record = warnings[0]
    assert record.request_id == request_id  # id is stamped on the record (drives the log prefix)
    assert "400" in record.getMessage()
    assert record.exc_info is not None  # traceback attached


def test_validation_error_logged_once_with_request_id(api_client, caplog):
    """A 422 validation error is logged once at warning level with the request_id and a traceback."""
    with caplog.at_level(logging.INFO):
        resp = api_client.post("/v1/audio/transcriptions")  # missing the required 'file' field

    assert resp.status_code == 422

    request_id = resp.headers["X-Request-ID"]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    record = warnings[0]
    assert record.request_id == request_id  # id is stamped on the record (drives the log prefix)
    assert "422" in record.getMessage()
    assert record.exc_info is not None  # traceback attached
