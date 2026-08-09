"""
AudioService — local Speech-to-Text (Whisper/MLX) and Text-to-Speech (Kokoro/MLX).

Design decisions
────────────────
- STT and TTS run *alongside* a loaded chat model, not in its place: the voice
  flow is STT → chat → TTS within a single turn. The whisper and Kokoro models
  are small (~tens to hundreds of MB) compared with the chat LLMs, so they keep
  their own resident handles instead of competing for the single large-model
  slot owned by InferenceService / MediaInferenceService.

- Both models load lazily on first use. `mlx_whisper.transcribe()` caches the
  loaded model internally keyed by repo, so STT needs no handle here and stays
  resident for the process lifetime. The Kokoro TTS model is loaded once and held
  under a lock; because we own that handle, it is dropped after
  ``tts_idle_timeout_seconds`` of inactivity (re-arming timer) to reclaim memory,
  then reloaded lazily on the next request.

- Loading never *downloads*. Both mlx-whisper and mlx-audio fall back to
  ``snapshot_download`` when handed a repo id they can't resolve locally, which
  turns a first request into a multi-GB stall. Every load is therefore gated on
  :meth:`AudioService._ensure_speech_model_available`, an offline cache probe;
  ``python -m app.cli.main audio prepare`` (``task audio:setup``) is the only path
  that fetches speech weights.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from app.config import Settings
from app.config import settings as global_settings
from app.core.exceptions import (
    InferenceError,
    InvalidVoiceError,
    MLXManagerError,
    SpeechModelNotPreparedError,
)
from app.core.logging import get_logger
from app.patches import patch_interpolate_ceil_drift

logger = get_logger(__name__)


class AudioService:
    """Lazily-loaded local STT (mlx-whisper) and TTS (mlx-audio / Kokoro)."""

    def __init__(self, cfg: Optional[Settings] = None) -> None:
        self._cfg = cfg or global_settings
        self._lock = threading.Lock()
        self._tts_model = None
        self._tts_model_name: Optional[str] = None
        self._tts_snapshot_path: Optional[Path] = None
        self._tts_idle_timer: Optional[threading.Timer] = None

    # ── Local availability ───────────────────────────────────────────────────

    def _ensure_speech_model_available(self, repo_id: str, label: str) -> Path:
        """
        Resolve a speech model's snapshot in the local HuggingFace cache.

        Purely filesystem — ``local_files_only=True`` never touches the network,
        so a missing model fails fast instead of downloading mid-request.

        Returns:
            Path to the cached snapshot directory.

        Raises:
            SpeechModelNotPreparedError: if the model is not in the local cache.
        """
        from huggingface_hub import snapshot_download

        try:
            return Path(snapshot_download(repo_id=repo_id, local_files_only=True))
        except Exception as exc:  # any local-resolution failure == not prepared
            raise SpeechModelNotPreparedError(label, repo_id) from exc

    # ── Speech-to-Text ───────────────────────────────────────────────────────

    def transcribe(self, audio_path: str | Path, language: Optional[str] = None) -> str:
        """
        Transcribe an audio file to text using Whisper on MLX.

        Non-WAV containers (e.g. the browser's webm/opus) are decoded via ffmpeg,
        which mlx-whisper shells out to internally.

        Raises:
            SpeechModelNotPreparedError: if the Whisper weights are not cached.
            InferenceError: if transcription fails.
        """
        # Hand mlx-whisper the resolved snapshot directory, not the repo id: its
        # loader short-circuits on an existing path, so it can never download.
        model_path = self._ensure_speech_model_available(self._cfg.stt_model, "STT (Whisper)")

        try:
            import mlx_whisper  # type: ignore

            result = mlx_whisper.transcribe(
                str(audio_path),
                path_or_hf_repo=str(model_path),
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
            # Probe outside the try below so a missing model surfaces as itself
            # rather than being rewrapped as a generic InferenceError.
            snapshot = self._ensure_speech_model_available(self._cfg.tts_model, "TTS (Kokoro)")
            try:
                self._configure_espeak()
                patch_interpolate_ceil_drift()  # see app/patches + PATCHES.md
                from mlx_audio.utils import load_model  # type: ignore

                logger.info("Loading TTS model '%s' …", self._cfg.tts_model)
                started_at = time.perf_counter()
                # Unlike whisper, mlx-audio must get the *repo id*: Kokoro's
                # config.json has no model_type, so the architecture is inferred
                # from the repo name — a snapshot-hash path breaks detection.
                self._tts_model = load_model(self._cfg.tts_model)
                self._tts_model_name = self._cfg.tts_model
                self._tts_snapshot_path = snapshot
                logger.info(
                    "TTS model '%s' loaded in %.2fs.",
                    self._cfg.tts_model,
                    time.perf_counter() - started_at,
                )
            except Exception as exc:
                self._tts_model = None
                self._tts_model_name = None
                self._tts_snapshot_path = None
                raise InferenceError(f"TTS model load failed: {exc}") from exc

    def _cancel_idle_timer(self) -> None:
        """Stop any pending idle-unload timer (called while a request is active)."""
        if self._tts_idle_timer is not None:
            self._tts_idle_timer.cancel()
            self._tts_idle_timer = None

    def _arm_idle_timer(self) -> None:
        """(Re)start the idle-unload countdown after a completed synthesis."""
        timeout = self._cfg.tts_idle_timeout_seconds
        if timeout <= 0:  # disabled → keep resident for the process lifetime
            return
        self._cancel_idle_timer()
        self._tts_idle_timer = threading.Timer(timeout, self._unload_tts_if_idle)
        self._tts_idle_timer.daemon = True
        self._tts_idle_timer.start()

    def _unload_tts_if_idle(self) -> None:
        """Drop the resident TTS model after the idle timeout elapses."""
        with self._lock:
            if self._tts_model is None:
                return
            logger.info(
                "Unloading idle TTS model '%s' after %.0fs.",
                self._tts_model_name,
                self._cfg.tts_idle_timeout_seconds,
            )
            self._tts_model = None
            self._tts_model_name = None
            self._tts_snapshot_path = None
            self._tts_idle_timer = None

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
            SpeechModelNotPreparedError: if the Kokoro weights are not cached, or
                no voice packs are cached at all.
            InvalidVoiceError: if the voice packs are cached but this voice isn't
                one of them — i.e. a bad name rather than a missing download.
            InferenceError: if the text is empty or synthesis fails.
        """
        if not text or not text.strip():
            raise InferenceError("input text is empty.")

        # Pause the idle-unload countdown so an in-flight request can't have the
        # model dropped out from under it; re-arm once synthesis completes.
        self._cancel_idle_timer()
        self._ensure_tts_loaded()

        try:
            # Kokoro fetches voice packs separately at generate() time, so an
            # uncached voice is a second way to trigger a mid-request download.
            voice_name = voice or self._cfg.tts_voice
            voices_dir = self._tts_snapshot_path / "voices"
            if not (voices_dir / f"{voice_name}.safetensors").exists():
                available = sorted(p.stem for p in voices_dir.glob("*.safetensors"))
                if available:  # the cache is prepared → the name is wrong, not the setup
                    raise InvalidVoiceError(voice_name, available)
                raise SpeechModelNotPreparedError(
                    "TTS (Kokoro)", self._cfg.tts_model, f" (missing voice pack '{voice_name}')"
                )

            chunks: list[np.ndarray] = []
            sample_rate = 24000  # Kokoro default; overwritten by the model below
            for result in self._tts_model.generate(
                text=text,
                voice=voice_name,
                speed=speed,
                lang_code=self._cfg.tts_lang_code,
            ):
                chunks.append(np.array(result.audio).astype(np.float32))
                sample_rate = result.sample_rate

            if not chunks:
                raise InferenceError("no audio was generated.")
            return np.concatenate(chunks), sample_rate
        except MLXManagerError:  # domain errors are already meaningful — don't rewrap
            raise
        except Exception as exc:
            raise InferenceError(f"speech synthesis failed: {exc}") from exc
        finally:
            self._arm_idle_timer()
