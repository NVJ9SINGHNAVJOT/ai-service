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

from app.core.exceptions import (
    InferenceError,
    InvalidLangCodeError,
    InvalidSTTModelError,
    InvalidVoiceError,
    SpeechModelNotPreparedError,
)


def test_speech_endpoint_returns_wav(api_client, monkeypatch):
    """POST /v1/audio/speech returns WAV audio bytes."""
    import app.main as main

    monkeypatch.setattr(
        main.audio_service,
        "synthesize",
        lambda text, voice=None, speed=1.0, lang_code=None: (np.zeros(2400, dtype=np.float32), 24000),
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

    def _boom(text, voice=None, speed=1.0, lang_code=None):
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
        lambda audio_path, language=None, model=None: "transcribed text",
    )

    files = {"file": ("clip.wav", io.BytesIO(b"RIFFfake-wav-bytes"), "audio/wav")}
    resp = api_client.post("/v1/audio/transcriptions", files=files)

    assert resp.status_code == 200
    assert resp.json() == {"text": "transcribed text"}


def test_speech_endpoint_returns_503_when_model_not_prepared(api_client, monkeypatch, caplog):
    """An unprepared TTS model is a 503 pointing at `task audio:setup`, logged once."""
    import app.main as main

    def _not_prepared(text, voice=None, speed=1.0, lang_code=None):
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

    def _bad_voice(text, voice=None, speed=1.0, lang_code=None):
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

    def _not_prepared(audio_path, language=None, model=None):
        raise SpeechModelNotPreparedError(
            "STT", "mlx-community/whisper-large-v3-turbo"
        )

    monkeypatch.setattr(main.audio_service, "transcribe", _not_prepared)

    files = {"file": ("clip.wav", io.BytesIO(b"RIFFfake-wav-bytes"), "audio/wav")}
    resp = api_client.post("/v1/audio/transcriptions", files=files)

    assert resp.status_code == 503
    assert "task audio:setup" in resp.json()["detail"]


def test_transcription_endpoint_requires_file(api_client):
    """Omitting the file field is a 422 validation error."""
    resp = api_client.post("/v1/audio/transcriptions")

    assert resp.status_code == 422


# ── per-request model / voice options ────────────────────────────────────────


def test_transcription_endpoint_forwards_model_field(api_client, monkeypatch):
    """The multipart `model` field selects the STT model instead of being ignored."""
    import app.main as main

    seen = {}

    def _transcribe(audio_path, language=None, model=None):
        seen["model"] = model
        seen["language"] = language
        return "ok"

    monkeypatch.setattr(main.audio_service, "transcribe", _transcribe)

    files = {"file": ("clip.wav", io.BytesIO(b"RIFFfake-wav-bytes"), "audio/wav")}
    resp = api_client.post(
        "/v1/audio/transcriptions",
        files=files,
        data={"model": "mlx-community/parakeet-tdt-0.6b-v2", "language": "en"},
    )

    assert resp.status_code == 200
    assert seen == {"model": "mlx-community/parakeet-tdt-0.6b-v2", "language": "en"}


def test_transcription_endpoint_returns_400_for_unknown_model(api_client, monkeypatch):
    """An unconfigured STT model is the client's error (400), not a 503."""
    import app.main as main

    def _bad_model(audio_path, language=None, model=None):
        raise InvalidSTTModelError("nope", ["mlx-community/whisper-large-v3-turbo"])

    monkeypatch.setattr(main.audio_service, "transcribe", _bad_model)

    files = {"file": ("clip.wav", io.BytesIO(b"RIFFfake-wav-bytes"), "audio/wav")}
    resp = api_client.post("/v1/audio/transcriptions", files=files, data={"model": "nope"})

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "nope" in detail and "whisper-large-v3-turbo" in detail
    assert "audio:setup" not in detail  # not a setup problem, so don't suggest setup


def test_speech_endpoint_forwards_lang_code(api_client, monkeypatch):
    """The request body's lang_code reaches the service."""
    import app.main as main

    seen = {}

    def _synthesize(text, voice=None, speed=1.0, lang_code=None):
        seen["lang_code"] = lang_code
        return np.zeros(2400, dtype=np.float32), 24000

    monkeypatch.setattr(main.audio_service, "synthesize", _synthesize)

    resp = api_client.post(
        "/v1/audio/speech", json={"input": "Hello.", "voice": "bf_emma", "lang_code": "b"}
    )

    assert resp.status_code == 200
    assert seen["lang_code"] == "b"


def test_speech_endpoint_returns_400_for_unknown_lang_code(api_client, monkeypatch):
    """An unsupported language code is a 400, alongside the unknown-voice case."""
    import app.main as main

    def _bad_lang(text, voice=None, speed=1.0, lang_code=None):
        raise InvalidLangCodeError("zz", ["a", "b"])

    monkeypatch.setattr(main.audio_service, "synthesize", _bad_lang)

    resp = api_client.post("/v1/audio/speech", json={"input": "hi", "lang_code": "zz"})

    assert resp.status_code == 400
    assert "zz" in resp.json()["detail"]


def test_audio_models_endpoint_describes_both_backends(api_client, monkeypatch):
    """GET /v1/audio/models gives a frontend everything it needs to call the other two."""
    import app.main as main

    monkeypatch.setattr(
        main.audio_service,
        "describe_stt",
        lambda: {
            "default": "mlx-community/whisper-large-v3-turbo",
            "models": [
                {
                    "id": "mlx-community/whisper-large-v3-turbo",
                    "backend": "mlx-whisper",
                    "ready": True,
                    "loaded": False,
                    "accepts_language_hint": True,
                    "languages": None,
                },
                {
                    "id": "mlx-community/parakeet-tdt-0.6b-v2",
                    "backend": "mlx-audio",
                    "ready": False,
                    "loaded": False,
                    "accepts_language_hint": False,
                    "languages": ["en"],
                },
            ],
        },
    )
    monkeypatch.setattr(
        main.audio_service,
        "describe_tts",
        lambda: {
            "model": "prince-canuma/Kokoro-82M",
            "ready": True,
            "loaded": False,
            "default_voice": "af_heart",
            "default_lang_code": "a",
            "voices": ["af_heart", "bf_emma"],
            "lang_codes": [{"code": "a", "label": "American English"}],
            "speed": {"min": 0.25, "max": 4.0, "default": 1.0},
            "response_formats": ["wav"],
        },
    )

    resp = api_client.get("/v1/audio/models")

    assert resp.status_code == 200
    body = resp.json()
    assert [m["id"] for m in body["stt"]["models"]] == [
        "mlx-community/whisper-large-v3-turbo",
        "mlx-community/parakeet-tdt-0.6b-v2",
    ]
    assert body["stt"]["models"][1]["ready"] is False  # UI can say "run audio:setup"
    assert body["tts"]["voices"] == ["af_heart", "bf_emma"]
    assert body["tts"]["response_formats"] == ["wav"]


def test_speech_500_logged_once_with_request_id_and_traceback(api_client, monkeypatch, caplog):
    """A synthesis failure (500) is logged exactly once, correlated by request_id, with a traceback."""
    import app.main as main

    def _boom(text, voice=None, speed=1.0, lang_code=None):
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
