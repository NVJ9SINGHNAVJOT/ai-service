"""
AudioService — local Speech-to-Text (Whisper / Parakeet on MLX) and
Text-to-Speech (Kokoro/MLX).

Design decisions
────────────────
- STT and TTS run *alongside* a loaded chat model, not in its place: the voice
  flow is STT → chat → TTS within a single turn. The speech models are small
  (~tens of MB to ~2.5 GB) compared with the chat LLMs, so they keep their own
  resident handles instead of competing for the single large-model slot owned by
  InferenceService / MediaInferenceService.

- Two independent slots (:class:`_ResidentModel`), one for STT and one for TTS.
  Each holds **one** model, loads it lazily on first use, and drops it after
  ``stt_idle_timeout_seconds`` / ``tts_idle_timeout_seconds`` of inactivity
  (re-arming timer, 0 = keep resident). A transcription request picks its model
  from ``settings.available_stt_models``; asking for a different one unloads the
  current one first, so at most one STT model is ever in memory.

- Two STT backends, chosen from the repo id. Whisper (``mlx-whisper``) is
  multilingual and takes a language hint; Parakeet TDT (``mlx-audio``) is faster
  and more accurate on English and doesn't hallucinate during silence, but takes
  no language hint. mlx-whisper caches its handle on its own module rather than
  handing us one, so the slot primes and clears
  ``mlx_whisper.transcribe.ModelHolder`` directly.

- Loading never *downloads*. Both mlx-whisper and mlx-audio fall back to
  ``snapshot_download`` when handed a repo id they can't resolve locally, which
  turns a first request into a multi-GB stall. Every load is therefore gated on
  :meth:`AudioService._ensure_speech_model_available`, an offline cache probe;
  ``python -m app.cli.main audio prepare`` (``task audio:setup``) is the only path
  that fetches speech weights.
"""

from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple

import numpy as np

from app.config import Settings
from app.config import settings as global_settings
from app.core.exceptions import (
    InferenceError,
    InvalidLangCodeError,
    InvalidSTTModelError,
    InvalidVoiceError,
    MLXManagerError,
    SpeechModelNotPreparedError,
)
from app.core.logging import get_logger
from app.patches import patch_interpolate_ceil_drift

logger = get_logger(__name__)

# Kokoro's language codes, in the order `GET /v1/audio/models` advertises them.
_KOKORO_LANG_CODES: dict[str, str] = {
    "a": "American English",
    "b": "British English",
    "e": "Spanish",
    "f": "French",
    "h": "Hindi",
    "i": "Italian",
    "j": "Japanese",
    "p": "Brazilian Portuguese",
    "z": "Mandarin Chinese",
}

# Playback-speed bounds advertised by `GET /v1/audio/models` — keep in sync with
# `SpeechRequest.speed` in app/schemas/audio.py, which enforces them.
_SPEED_MIN, _SPEED_MAX, _SPEED_DEFAULT = 0.25, 4.0, 1.0

# Long inputs are transcribed in overlapping windows: mlx-audio's Parakeet
# defaults to no chunking, which would hold an hour-long file in memory at once.
_PARAKEET_CHUNK_SECONDS = 120.0
_PARAKEET_OVERLAP_SECONDS = 15.0


def _stt_backend_for(repo_id: str) -> Optional[str]:
    """
    Map an STT repo id to the package that can load it, or None if neither can.

    Repo ids are the only signal available before download, and both families
    name themselves in the repo (``…/whisper-large-v3-turbo``,
    ``…/parakeet-tdt-0.6b-v2``).
    """
    name = repo_id.lower()
    if "parakeet" in name:
        return "mlx-audio"
    if "whisper" in name:
        return "mlx-whisper"
    return None


class _ResidentModel:
    """
    One in-memory model slot with a re-arming idle-unload timer.

    Holds at most one model, and requests for **that** model run concurrently:
    the lock guards load/unload only, so generation happens outside it.
    Requesting a *different* model waits for the in-flight ones to drain before
    swapping — evicting underneath them would pull the handle out from under a
    running generate (for Whisper, the shared `ModelHolder` it is about to read).

    The in-flight counter serves both that wait and the idle timer, which must
    not drop a model mid-request; ``Timer.cancel()`` alone can't guarantee that,
    since it's a no-op once the callback has already started running.
    """

    def __init__(
        self,
        label: str,
        idle_timeout: Callable[[], float],
        unload_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        self._label = label
        self._idle_timeout = idle_timeout
        self._unload_hook = unload_hook
        self._cond = threading.Condition(threading.RLock())
        self._model = None
        self._name: Optional[str] = None
        self._snapshot: Optional[Path] = None
        self._timer: Optional[threading.Timer] = None
        self._in_flight = 0

    @property
    def loaded_name(self) -> Optional[str]:
        """The resident model's name, or None when the slot is empty."""
        return self._name

    @contextmanager
    def acquire(
        self, name: str, loader: Callable[[], Tuple[object, Path]]
    ) -> Iterator[Tuple[object, Path]]:
        """
        Yield ``(handle, snapshot_path)`` for `name`, loading it if needed.

        `loader` returns the pair to cache; it runs under the lock and only when
        the slot doesn't already hold `name`. Concurrent callers asking for the
        model already resident proceed straight through; one asking for a
        different model blocks until the others have finished.
        """
        with self._cond:
            # A different model is wanted but this one is still generating —
            # wait it out rather than yanking the handle away mid-request.
            while self._name is not None and self._name != name and self._in_flight:
                self._cond.wait()
            self._cancel_timer()  # after the wait: the drain re-armed it
            if self._name != name:
                self._release()
            if self._model is None:
                logger.info("Loading %s model '%s' …", self._label, name)
                started_at = time.perf_counter()
                self._model, self._snapshot = loader()
                self._name = name
                logger.info(
                    "%s model '%s' loaded in %.2fs.",
                    self._label,
                    name,
                    time.perf_counter() - started_at,
                )
            self._in_flight += 1
            model, snapshot = self._model, self._snapshot
        try:
            yield model, snapshot  # generation runs outside the lock
        finally:
            with self._cond:
                self._in_flight -= 1
                self._arm_timer()
                self._cond.notify_all()  # a swap may be waiting on the drain

    def _release(self) -> None:
        """Drop the resident handle. Caller holds the lock."""
        if self._unload_hook is not None:
            self._unload_hook()
        self._model = None
        self._name = None
        self._snapshot = None

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _arm_timer(self) -> None:
        """(Re)start the idle-unload countdown after a completed request."""
        timeout = self._idle_timeout()
        if timeout <= 0:  # disabled → keep resident for the process lifetime
            return
        self._cancel_timer()
        self._timer = threading.Timer(timeout, self._unload_if_idle)
        self._timer.daemon = True
        self._timer.start()

    def _unload_if_idle(self) -> None:
        """Drop the resident model once the idle timeout elapses."""
        with self._cond:
            self._timer = None
            if self._model is None or self._in_flight:
                return
            logger.info(
                "Unloading idle %s model '%s' after %.0fs.",
                self._label,
                self._name,
                self._idle_timeout(),
            )
            self._release()


class AudioService:
    """Lazily-loaded local STT (mlx-whisper / mlx-audio) and TTS (mlx-audio / Kokoro)."""

    def __init__(self, cfg: Optional[Settings] = None) -> None:
        self._cfg = cfg or global_settings
        self._stt = _ResidentModel(
            "STT", lambda: self._cfg.stt_idle_timeout_seconds, _release_whisper_cache
        )
        self._tts = _ResidentModel("TTS", lambda: self._cfg.tts_idle_timeout_seconds)

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

    def _probe(self, repo_id: str) -> Optional[Path]:
        """Non-raising variant of the cache probe, for the read-only describe calls."""
        try:
            return self._ensure_speech_model_available(repo_id, "")  # label unused here
        except SpeechModelNotPreparedError:
            return None

    # ── Capability description (drives GET /v1/audio/models) ─────────────────

    def describe_stt(self) -> dict:
        """Everything a client needs to choose an STT model. Loads nothing."""
        loaded = self._stt.loaded_name
        models = []
        for repo in self._cfg.available_stt_models:
            backend = _stt_backend_for(repo)
            languages = None  # None = auto-detected or not enumerated
            if backend == "mlx-audio" and "v2" in repo.lower():
                languages = ["en"]  # Parakeet TDT v2 is English-only
            models.append(
                {
                    "id": repo,
                    "backend": backend,
                    "ready": self._probe(repo) is not None,
                    "loaded": repo == loaded,
                    "accepts_language_hint": backend == "mlx-whisper",
                    "languages": languages,
                }
            )
        return {"default": self._cfg.stt_model, "models": models}

    def describe_tts(self) -> dict:
        """Everything a client needs to call /v1/audio/speech. Loads nothing."""
        snapshot = self._probe(self._cfg.tts_model)
        voices = (
            sorted(p.stem for p in (snapshot / "voices").glob("*.safetensors"))
            if snapshot is not None
            else []
        )
        return {
            "model": self._cfg.tts_model,
            # Voice packs are downloaded separately from the weights, and
            # synthesis fails without them — so both are part of "ready".
            "ready": snapshot is not None and bool(voices),
            "loaded": self._tts.loaded_name is not None,
            "default_voice": self._cfg.tts_voice,
            "default_lang_code": self._cfg.tts_lang_code,
            "voices": voices,
            "lang_codes": [
                {"code": code, "label": label} for code, label in _KOKORO_LANG_CODES.items()
            ],
            "speed": {"min": _SPEED_MIN, "max": _SPEED_MAX, "default": _SPEED_DEFAULT},
            "response_formats": ["wav"],
        }

    # ── Speech-to-Text ───────────────────────────────────────────────────────

    def _resolve_stt_model(self, requested: Optional[str]) -> str:
        """
        Pick the STT repo for this request.

        The identifier is the HuggingFace repo id verbatim — the ``org__name``
        sanitizing is on-disk naming for `models/downloaded` and `models/custom`
        only, and has no meaning here.
        """
        if not requested or not requested.strip():
            return self._cfg.stt_model
        repo = requested.strip()
        available = self._cfg.available_stt_models
        if repo not in available:
            raise InvalidSTTModelError(repo, available)
        return repo

    def _load_whisper(self, repo_id: str) -> Tuple[object, Path]:
        """Prime mlx-whisper's module-level handle so our slot's timer governs it."""
        snapshot = self._ensure_speech_model_available(repo_id, "STT")
        try:
            import mlx.core as mx
            from mlx_whisper.transcribe import ModelHolder  # type: ignore

            # float16 matches transcribe()'s own default (fp16=True), so the
            # transcribe call below reuses this handle instead of reloading.
            return ModelHolder.get_model(str(snapshot), mx.float16), snapshot
        except Exception as exc:
            raise InferenceError(f"STT model load failed: {exc}") from exc

    def _load_parakeet(self, repo_id: str) -> Tuple[object, Path]:
        """Load a Parakeet model from its resolved snapshot directory."""
        snapshot = self._ensure_speech_model_available(repo_id, "STT")
        try:
            from mlx_audio.stt.utils import load_model  # type: ignore

            # Parakeet ships a raw NeMo config.json with no `model_type`, so
            # mlx-audio would fall back to guessing the architecture from the
            # directory name — which here is an opaque snapshot hash. Naming it
            # explicitly lets us load by path, where a download is impossible.
            return load_model(str(snapshot), model_type="parakeet"), snapshot
        except Exception as exc:
            raise InferenceError(f"STT model load failed: {exc}") from exc

    def transcribe(
        self,
        audio_path: str | Path,
        language: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Transcribe an audio file to text.

        Args:
            audio_path: File to transcribe. Non-WAV containers (e.g. the
                browser's webm/opus) are decoded via ffmpeg by both backends.
            language: ISO-639-1 hint. Honored by Whisper; ignored by Parakeet,
                which takes no such argument.
            model: HF repo id from ``settings.available_stt_models``; the
                configured default when omitted.

        Raises:
            InvalidSTTModelError: if `model` isn't one of the configured repos.
            SpeechModelNotPreparedError: if its weights are not cached.
            InferenceError: if loading or transcription fails.
        """
        repo_id = self._resolve_stt_model(model)
        backend = _stt_backend_for(repo_id)
        if backend is None:
            raise InferenceError(
                f"no STT backend for '{repo_id}': the repo id must name either "
                "'whisper' or 'parakeet'."
            )

        loader = self._load_parakeet if backend == "mlx-audio" else self._load_whisper
        try:
            with self._stt.acquire(repo_id, lambda: loader(repo_id)) as (handle, snapshot):
                if backend == "mlx-audio":
                    if language:
                        logger.debug(
                            "Ignoring language hint '%s': %s takes none.", language, repo_id
                        )
                    result = handle.generate(
                        str(audio_path),
                        chunk_duration=_PARAKEET_CHUNK_SECONDS,
                        overlap_duration=_PARAKEET_OVERLAP_SECONDS,
                    )
                    return (result.text or "").strip()

                import mlx_whisper  # type: ignore

                # Hand mlx-whisper the resolved snapshot directory, not the repo
                # id: its loader short-circuits on an existing path, so it can
                # never download — and it hits the handle primed above.
                result = mlx_whisper.transcribe(
                    str(audio_path),
                    path_or_hf_repo=str(snapshot),
                    language=language,
                )
                return (result.get("text") or "").strip()
        except MLXManagerError:  # domain errors are already meaningful — don't rewrap
            raise
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

    def _ensure_tts_loaded(self) -> Tuple[object, Path]:
        """Load the Kokoro TTS model (the TTS slot's loader — see PATCHES.md)."""
        snapshot = self._ensure_speech_model_available(self._cfg.tts_model, "TTS (Kokoro)")
        try:
            self._configure_espeak()
            patch_interpolate_ceil_drift()  # see app/patches + PATCHES.md
            from mlx_audio.utils import load_model  # type: ignore

            # Unlike whisper, mlx-audio must get the *repo id*: Kokoro's
            # config.json has no model_type, so the architecture is inferred
            # from the repo name — a snapshot-hash path breaks detection.
            return load_model(self._cfg.tts_model), snapshot
        except Exception as exc:
            raise InferenceError(f"TTS model load failed: {exc}") from exc

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        lang_code: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from text.

        Returns:
            Tuple of (mono float32 PCM samples, sample_rate).

        Raises:
            InvalidLangCodeError: if `lang_code` isn't one Kokoro supports.
            SpeechModelNotPreparedError: if the Kokoro weights are not cached, or
                no voice packs are cached at all.
            InvalidVoiceError: if the voice packs are cached but this voice isn't
                one of them — i.e. a bad name rather than a missing download.
            InferenceError: if the text is empty or synthesis fails.
        """
        if not text or not text.strip():
            raise InferenceError("input text is empty.")

        lang = (lang_code or self._cfg.tts_lang_code).strip()
        if lang not in _KOKORO_LANG_CODES:
            raise InvalidLangCodeError(lang, list(_KOKORO_LANG_CODES))

        try:
            with self._tts.acquire(self._cfg.tts_model, self._ensure_tts_loaded) as (
                tts_model,
                snapshot,
            ):
                # Kokoro fetches voice packs separately at generate() time, so an
                # uncached voice is a second way to trigger a mid-request download.
                voice_name = voice or self._cfg.tts_voice
                voices_dir = snapshot / "voices"
                if not (voices_dir / f"{voice_name}.safetensors").exists():
                    available = sorted(p.stem for p in voices_dir.glob("*.safetensors"))
                    if available:  # the cache is prepared → the name is wrong, not the setup
                        raise InvalidVoiceError(voice_name, available)
                    raise SpeechModelNotPreparedError(
                        "TTS (Kokoro)",
                        self._cfg.tts_model,
                        f" (missing voice pack '{voice_name}')",
                    )

                chunks: list[np.ndarray] = []
                sample_rate = 24000  # Kokoro default; overwritten by the model below
                for result in tts_model.generate(
                    text=text,
                    voice=voice_name,
                    speed=speed,
                    lang_code=lang,
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


def _release_whisper_cache() -> None:
    """
    Drop mlx-whisper's module-level model handle.

    mlx-whisper memoizes the loaded model on ``ModelHolder`` rather than handing
    us a handle, so the STT slot has to clear it there. A no-op when whisper was
    never imported (e.g. only Parakeet has been used), which also keeps the
    import out of the unload path.
    """
    module = sys.modules.get("mlx_whisper.transcribe")
    if module is None:
        return
    module.ModelHolder.model = None
    module.ModelHolder.model_path = None
