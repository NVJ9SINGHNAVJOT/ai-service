"""
AudioService — local Speech-to-Text (Whisper/MLX) and Text-to-Speech (Kokoro/MLX).

Design decisions
────────────────
- STT and TTS run *alongside* a loaded chat model, not in its place: the voice
  flow is STT → chat → TTS within a single turn. The whisper and Kokoro models
  are small (~tens to hundreds of MB) compared with the chat LLMs, so they keep
  their own resident handles instead of competing for the single large-model
  slot owned by InferenceService / MediaInferenceService.

- Both models load lazily on first use and are then cached for the process
  lifetime. `mlx_whisper.transcribe()` caches the loaded model internally keyed
  by repo, so STT needs no handle here; the Kokoro TTS model is loaded once and
  held under a lock.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from app.config import Settings
from app.config import settings as global_settings
from app.core.exceptions import InferenceError
from app.core.logging import get_logger

logger = get_logger(__name__)


class AudioService:
    """Lazily-loaded local STT (mlx-whisper) and TTS (mlx-audio / Kokoro)."""

    def __init__(self, cfg: Optional[Settings] = None) -> None:
        self._cfg = cfg or global_settings
        self._lock = threading.Lock()
        self._tts_model = None
        self._tts_model_name: Optional[str] = None

    # ── Speech-to-Text ───────────────────────────────────────────────────────

    def transcribe(self, audio_path: str | Path, language: Optional[str] = None) -> str:
        """
        Transcribe an audio file to text using Whisper on MLX.

        Non-WAV containers (e.g. the browser's webm/opus) are decoded via ffmpeg,
        which mlx-whisper shells out to internally.

        Raises:
            InferenceError: if transcription fails.
        """
        try:
            import mlx_whisper  # type: ignore

            result = mlx_whisper.transcribe(
                str(audio_path),
                path_or_hf_repo=self._cfg.stt_model,
                language=language,
            )
            return (result.get("text") or "").strip()
        except Exception as exc:
            raise InferenceError(f"transcription failed: {exc}") from exc

    # ── Text-to-Speech ───────────────────────────────────────────────────────

    @staticmethod
    def _configure_espeak() -> None:
        """
        Point phonemizer at the bundled espeak-ng library/data.

        Kokoro's misaki front-end uses espeak-ng to phonemize out-of-dictionary
        words (numbers, acronyms, names — common in LLM replies). The
        ``espeakng-loader`` wheel ships the library so no system install is
        needed. Best-effort: if this fails, Kokoro still handles in-dictionary
        words, just skipping unknown ones.
        """
        try:
            import espeakng_loader
            from phonemizer.backend.espeak.wrapper import EspeakWrapper

            EspeakWrapper.set_library(espeakng_loader.get_library_path())
            EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
            espeakng_loader.make_library_available()
        except Exception as exc:  # noqa: BLE001 — degraded mode is acceptable
            logger.warning("espeak-ng fallback unavailable (%s); OOD words may be skipped.", exc)

    def _ensure_tts_loaded(self) -> None:
        """Load and cache the Kokoro TTS model on first use (thread-safe)."""
        if self._tts_model is not None:
            return
        with self._lock:
            if self._tts_model is not None:
                return
            try:
                self._configure_espeak()
                from mlx_audio.utils import load_model  # type: ignore

                logger.info("Loading TTS model '%s' …", self._cfg.tts_model)
                started_at = time.perf_counter()
                self._tts_model = load_model(self._cfg.tts_model)
                self._tts_model_name = self._cfg.tts_model
                logger.info(
                    "TTS model '%s' loaded in %.2fs.",
                    self._cfg.tts_model,
                    time.perf_counter() - started_at,
                )
            except Exception as exc:
                self._tts_model = None
                self._tts_model_name = None
                raise InferenceError(f"TTS model load failed: {exc}") from exc

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from text.

        Returns:
            Tuple of (mono float32 PCM samples, sample_rate).

        Raises:
            InferenceError: if the text is empty or synthesis fails.
        """
        if not text or not text.strip():
            raise InferenceError("input text is empty.")

        self._ensure_tts_loaded()

        try:
            chunks: list[np.ndarray] = []
            sample_rate = 24000  # Kokoro default; overwritten by the model below
            for result in self._tts_model.generate(
                text=text,
                voice=voice or self._cfg.tts_voice,
                speed=speed,
                lang_code=self._cfg.tts_lang_code,
            ):
                chunks.append(np.array(result.audio).astype(np.float32))
                sample_rate = result.sample_rate

            if not chunks:
                raise InferenceError("no audio was generated.")
            return np.concatenate(chunks), sample_rate
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(f"speech synthesis failed: {exc}") from exc
