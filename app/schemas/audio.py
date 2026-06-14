"""
Pydantic schemas for the audio (STT / TTS) endpoints.

These mirror the OpenAI audio API shapes so the Java gateway can call them
through the same OpenAI-compatible plumbing it already uses for chat.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class SpeechRequest(BaseModel):
    """Request body for ``POST /v1/audio/speech`` (OpenAI-compatible subset)."""

    # Unknown OpenAI fields (e.g. `instructions`) are accepted but ignored so
    # callers using the OpenAI SDK don't get rejected.
    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "example": {
                "input": "Hello! This is a local text to speech test.",
                "voice": "af_heart",
            }
        },
    )

    input: str = Field(..., description="The text to synthesize into speech.")
    voice: Optional[str] = Field(
        default=None,
        description="Kokoro voice id (e.g. 'af_heart'). Falls back to the server default when omitted.",
    )
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Playback speed multiplier.")
    response_format: str = Field(
        default="wav",
        description="Output container. Only 'wav' is supported by this local endpoint.",
    )
    # Accepted for OpenAI compatibility; the TTS model is fixed server-side.
    model: Optional[str] = None


class TranscriptionResponse(BaseModel):
    """Response body for ``POST /v1/audio/transcriptions`` (OpenAI 'json' format)."""

    text: str
