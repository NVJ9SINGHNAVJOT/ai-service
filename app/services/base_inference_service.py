"""
Shared base class for the model-backed inference services.

`InferenceService` (mlx-lm, text-only) and `MediaInferenceService` (mlx-vlm,
multimodal) both keep a single model resident in memory and expose the same
lifecycle: ``load`` → ``chat`` / ``chat_stream`` → ``unload``. The bookkeeping
around that lifecycle — the reentrant lock, the loaded-model name, the load
duration, and the cross-process "running" marker — is identical between them.

Centralising it here keeps the two backends DRY and gives the OpenAI-compatible
route a single, substitutable interface (Liskov): the route picks whichever
service matches the request and calls the same methods regardless of backend.
Backend-specific behaviour is delegated to subclasses through the abstract
``load`` / ``chat`` / ``chat_stream`` methods and the ``_release_backend`` hook
(Template Method).
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple

from app.config import Settings, settings as _default_settings
from app.core.exceptions import InferenceError
from app.schemas.inference import ChatMessage, Role
from app.services.model_runtime_state import ModelRuntimeState


class LoadedModelService(ABC):
    """
    Manage a single in-memory model and the state around its lifecycle.

    Subclasses own the backend specifics (which library loads the model and how
    inference runs); this base owns the shared, backend-agnostic bookkeeping.

    Thread-safety: a reentrant lock protects load/unload. Concurrent generate
    calls on the same loaded model are NOT safe (MLX is not thread-safe at the
    C level); for a multi-user server you would add a request queue.
    """

    #: Error surfaced when inference is requested before a model is loaded.
    _NOT_LOADED_MESSAGE = "No model is currently loaded. Call load() first."

    def __init__(self, cfg: Optional[Settings] = None) -> None:
        self._cfg = cfg or _default_settings
        self._lock = threading.RLock()
        self._loaded_name: Optional[str] = None
        self._model: Optional[Any] = None
        self._last_load_duration_s: Optional[float] = None
        self._runtime_state = ModelRuntimeState(cfg=self._cfg)
        self._running_marker: Optional[Path] = None

    # ── Shared lifecycle state ───────────────────────────────────────────────

    @property
    def loaded_model_name(self) -> Optional[str]:
        """Name of the currently loaded model, or None."""
        return self._loaded_name

    @property
    def is_loaded(self) -> bool:
        """Return True when a model is resident in memory."""
        return self._model is not None

    @property
    def last_load_duration_s(self) -> Optional[float]:
        """Duration of the most recent successful model load, in seconds."""
        return self._last_load_duration_s

    def unload(self) -> Optional[str]:
        """
        Unload the currently loaded model and free memory.

        Returns:
            The name of the model that was unloaded, or None if nothing was loaded.
        """
        with self._lock:
            name = self._loaded_name
            self._unload_internal()
            return name

    def _require_loaded(self) -> None:
        """Raise a consistent error if inference is requested before load()."""
        if not self.is_loaded:
            raise InferenceError(self._NOT_LOADED_MESSAGE)

    def _unload_internal(self) -> None:
        """
        Internal unload without acquiring the lock (caller must hold it).

        Clears the runtime marker and the shared references, deferring to the
        subclass to release any backend-specific handles. MLX manages its own
        memory pool; dropping all Python references lets the GC reclaim memory.
        """
        self._runtime_state.clear_marker(self._running_marker)
        self._running_marker = None
        self._release_backend()
        self._model = None
        self._loaded_name = None
        self._last_load_duration_s = None

    def _generation_kwargs(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> dict:
        """Build the MLX generation keyword arguments shared by both backends."""
        max_tokens = max_tokens or self._cfg.default_max_tokens
        temperature = temperature if temperature is not None else self._cfg.default_temperature
        top_p = top_p if top_p is not None else self._cfg.default_top_p
        rep_pen = repetition_penalty if repetition_penalty is not None else self._cfg.default_repetition_penalty

        from mlx_lm.sample_utils import make_logits_processors, make_sampler  # type: ignore

        return {
            "max_tokens": max_tokens,
            "sampler": make_sampler(temp=temperature, top_p=top_p),
            "logits_processors": make_logits_processors(repetition_penalty=rep_pen),
        }

    # ── Backend-specific hooks (subclass responsibility) ─────────────────────

    @abstractmethod
    def _release_backend(self) -> None:
        """Drop backend-specific handles (tokenizer/processor/etc.) on unload."""

    @abstractmethod
    def load(self, model_path: Path, model_name: str) -> None:
        """Load a model from the given local path, swapping out any current one."""

    @abstractmethod
    def chat(
        self,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> Tuple[str, dict]:
        """Run one buffered chat completion and return ``(text, usage)``."""

    @abstractmethod
    def chat_stream(
        self,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> Generator[Tuple[str, Optional[dict]], None, None]:
        """Stream chat completion chunks as ``(text_chunk, usage_or_none)``."""


def openai_role_for_template(role: Role) -> str:
    """Map OpenAI roles onto the smaller role set most local chat templates expect."""
    if role == Role.developer:
        return Role.system.value
    return role.value
