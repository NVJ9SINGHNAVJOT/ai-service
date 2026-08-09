"""
Pydantic schemas for the audio (STT / TTS) endpoints.

These mirror the OpenAI audio API shapes so the Java gateway can call them
through the same OpenAI-compatible plumbing it already uses for chat.
"""

from __future__ import annotations

from typing import List, Optional

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
    lang_code: Optional[str] = Field(
        default=None,
        description=(
            "Kokoro language code — one of the `lang_codes` from GET /v1/audio/models "
            "(e.g. 'a' = American English, 'b' = British). Falls back to the server "
            "default when omitted. Should match the voice's prefix letter."
        ),
    )
    response_format: str = Field(
        default="wav",
        description="Output container. Only 'wav' is supported by this local endpoint.",
    )
    # Accepted for OpenAI compatibility; the TTS model is fixed server-side.
    model: Optional[str] = None


class TranscriptionResponse(BaseModel):
    """Response body for ``POST /v1/audio/transcriptions`` (OpenAI 'json' format)."""

    text: str


class STTModelInfo(BaseModel):
    """One selectable speech-to-text model."""

    id: str = Field(..., description="HuggingFace repo id — pass this as the `model` form field.")
    backend: Optional[str] = Field(
        default=None, description="Loader package: 'mlx-whisper' or 'mlx-audio'."
    )
    ready: bool = Field(..., description="Whether the weights are in the local cache.")
    loaded: bool = Field(..., description="Whether this is the currently resident STT model.")
    accepts_language_hint: bool = Field(
        ..., description="Whether the `language` form field has any effect for this model."
    )
    languages: Optional[List[str]] = Field(
        default=None,
        description="Supported languages, or null when auto-detected / not enumerated.",
    )


class STTInfo(BaseModel):
    """Speech-to-text capabilities."""

    default: str = Field(..., description="Model used when a request omits `model`.")
    models: List[STTModelInfo]


class LangCodeInfo(BaseModel):
    """A TTS language code and its human-readable name."""

    code: str
    label: str


class SpeedRange(BaseModel):
    """Accepted range for `SpeechRequest.speed`."""

    min: float
    max: float
    default: float


class TTSInfo(BaseModel):
    """Text-to-speech capabilities."""

    model: str
    ready: bool = Field(..., description="Whether the weights *and* voice packs are cached.")
    loaded: bool = Field(..., description="Whether the model is currently resident.")
    default_voice: str
    default_lang_code: str
    voices: List[str] = Field(..., description="Voice ids available in the local cache.")
    lang_codes: List[LangCodeInfo]
    speed: SpeedRange
    response_formats: List[str]


class AudioCapabilitiesResponse(BaseModel):
    """Response body for ``GET /v1/audio/models``."""

    stt: STTInfo
    tts: TTSInfo
