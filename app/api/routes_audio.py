"""
Audio API routes — local, OpenAI-compatible STT and TTS.

- POST /v1/audio/transcriptions  (multipart audio  → {"text": ...})
- POST /v1/audio/speech          (JSON {input,...}  → audio/wav bytes)

Both run fully on-device (Whisper + Kokoro on MLX); no audio or text leaves the
machine. The actual model work lives in :class:`AudioService`.
"""

from __future__ import annotations

import io
import os
import tempfile
from typing import Optional

import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from app.core.exceptions import InferenceError, InvalidVoiceError, SpeechModelNotPreparedError
from app.core.logging import get_logger
from app.schemas.audio import SpeechRequest, TranscriptionResponse
from app.api.response import send_response

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/audio", tags=["audio"])


@router.post(
    "/transcriptions",
    response_model=TranscriptionResponse,
    summary="Transcribe speech to text",
)
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(..., description="Audio file to transcribe (wav, mp3, webm, …)."),
    model: Optional[str] = Form(None, description="Accepted for OpenAI compatibility; ignored."),
    language: Optional[str] = Form(None, description="Optional ISO-639-1 language hint; auto-detected when omitted."),
    response_format: Optional[str] = Form(None, description="Accepted for OpenAI compatibility; always returns JSON."),
):
    """Transcribe an uploaded audio clip with Whisper on MLX."""
    from app.main import audio_service  # local import to avoid circular import

    suffix = os.path.splitext(file.filename or "")[1] or ".webm"
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        text = audio_service.transcribe(tmp_path, language=language)
    except SpeechModelNotPreparedError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except InferenceError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return send_response(request, TranscriptionResponse(text=text))


@router.post(
    "/speech",
    summary="Synthesize text to speech",
    responses={200: {"content": {"audio/wav": {}}, "description": "WAV audio bytes."}},
)
async def create_speech(request: Request, body: SpeechRequest) -> Response:
    """Synthesize speech from text with Kokoro on MLX and return WAV bytes."""
    from app.main import audio_service  # local import to avoid circular import

    if body.response_format and body.response_format.lower() != "wav":
        raise HTTPException(
            status_code=400,
            detail="Field 'response_format' is only supported with 'wav' on this local endpoint.",
        )

    try:
        audio, sample_rate = audio_service.synthesize(body.input, voice=body.voice, speed=body.speed)

        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV")
        buffer.seek(0)
    except InvalidVoiceError as exc:  # a bad name, not a missing download
        raise HTTPException(status_code=400, detail=str(exc))
    except SpeechModelNotPreparedError as exc:  # before the catch-all below
        raise HTTPException(status_code=503, detail=str(exc))
    except InferenceError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"speech synthesis failed: {exc}")

    request_id = getattr(request.state, "request_id", "unknown")
    logger.info("Response sent | request_id=%s | %d bytes audio/wav", request_id, buffer.getbuffer().nbytes)
    return Response(content=buffer.read(), media_type="audio/wav")
