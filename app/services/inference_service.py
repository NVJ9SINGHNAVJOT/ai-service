"""
InferenceService — loads an MLX-LM model into memory and runs inference.

Design decisions
────────────────
- A single model is kept in memory at a time (singleton cache).  Loading a
  second model automatically unloads the first.  This is the right trade-off
  for a local Mac server with limited unified memory.

- We use mlx_lm.load() to load the model+tokenizer, and mlx_lm.generate()
  to run inference.  Both are the officially supported Python APIs.

- We use both mlx_lm.generate() for buffered responses and
  mlx_lm.stream_generate() for token streaming.

- Chat / instruction models expect messages formatted with the tokenizer's
  chat template (tokenizer.apply_chat_template()).  We apply the template
  when the tokenizer supports it; otherwise we fall back to a simple
  "System: ...\nUser: ...\nAssistant:" format.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple

from app.config import Settings, settings as _default_settings
from app.core.exceptions import InferenceError, ModelLoadError
from app.core.logging import get_logger
from app.schemas.inference import ChatMessage, Role
from app.services.model_runtime_state import ModelRuntimeState

logger = get_logger(__name__)


class InferenceService:
    """
    Manages a single loaded MLX-LM model and exposes inference methods.

    Thread-safety: a reentrant lock protects model load/unload operations.
    Concurrent generate() calls on the same loaded model are NOT safe
    (MLX is not thread-safe at the C level).  For a production multi-user
    server you would add a request queue; this is fine for local single-user
    use.
    """

    def __init__(self, cfg: Optional[Settings] = None) -> None:
        self._cfg = cfg or _default_settings
        self._lock = threading.RLock()
        self._loaded_name: Optional[str] = None
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._last_load_duration_s: Optional[float] = None
        self._runtime_state = ModelRuntimeState(cfg=self._cfg)
        self._running_marker: Optional[Path] = None

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def loaded_model_name(self) -> Optional[str]:
        """Name of the currently loaded model, or None."""
        return self._loaded_name

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def last_load_duration_s(self) -> Optional[float]:
        """Duration of the most recent successful model load, in seconds."""
        return self._last_load_duration_s

    # ── Load / unload ────────────────────────────────────────────────────────

    def load(self, model_path: Path, model_name: str) -> None:
        """
        Load a model from the given local path.

        If a different model is already loaded it will be unloaded first.
        If the requested model is already loaded, this is a no-op.

        Args:
            model_path: Absolute path to the model directory.
            model_name: Human-readable name used for logging and status.

        Raises:
            ModelLoadError: if mlx_lm.load() fails.
        """
        with self._lock:
            if self._loaded_name == model_name:
                if self._running_marker is None:
                    self._running_marker = self._runtime_state.mark_running(model_name)
                logger.debug("Model '%s' is already loaded.", model_name)
                return

            if self._model is not None:
                logger.info("Unloading current model '%s' to load '%s'.", self._loaded_name, model_name)
                self._unload_internal()

            logger.info("Loading model '%s' from %s …", model_name, model_path)
            try:
                from mlx_lm import load as mlx_load  # type: ignore
                started_at = time.perf_counter()
                self._model, self._tokenizer = mlx_load(str(model_path))
                self._last_load_duration_s = time.perf_counter() - started_at
                self._loaded_name = model_name
                self._running_marker = self._runtime_state.mark_running(model_name)
                logger.info("Model '%s' loaded successfully.", model_name)
            except Exception as exc:
                self._model = None
                self._tokenizer = None
                self._loaded_name = None
                self._last_load_duration_s = None
                self._runtime_state.clear_marker(self._running_marker)
                self._running_marker = None
                raise ModelLoadError(model_name, str(exc)) from exc

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
            raise InferenceError("No model is currently loaded. Call load() first.")

    def _unload_internal(self) -> None:
        """Internal unload without acquiring the lock (caller must hold it)."""
        self._runtime_state.clear_marker(self._running_marker)
        self._running_marker = None
        self._model = None
        self._tokenizer = None
        self._loaded_name = None
        self._last_load_duration_s = None
        # MLX manages its own memory pool; there is no explicit free() call.
        # Removing all Python references allows the GC to release the memory.

    # ── Inference ────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> Tuple[str, dict]:
        """
        Generate text completion for the given prompt.

        Args:
            prompt:             Raw text prompt.
            max_tokens:         Max new tokens to generate.
            temperature:        Sampling temperature.
            top_p:              Nucleus sampling threshold.
            repetition_penalty: Repetition penalty factor.

        Returns:
            Tuple of (generated_text, usage_dict).

        Raises:
            InferenceError: if no model is loaded or generation fails.
        """
        with self._lock:
            self._require_loaded()

            try:
                from mlx_lm import generate as mlx_generate  # type: ignore

                result = mlx_generate(
                    self._model,
                    self._tokenizer,
                    prompt=prompt,
                    verbose=False,
                    **self._generation_kwargs(
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                    ),
                )
                text: str = result if isinstance(result, str) else str(result)
                return text, {}
            except Exception as exc:
                raise InferenceError(str(exc)) from exc

    def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> Generator[Tuple[str, Optional[dict]], None, None]:
        """
        Stream text completion chunks for the given prompt.

        Yields:
            Tuples of ``(text_chunk, usage_dict_or_none)``. The final yielded item
            includes usage metadata when available.
        """
        with self._lock:
            self._require_loaded()

            try:
                from mlx_lm import stream_generate as mlx_stream_generate  # type: ignore

                started_at = time.perf_counter()
                first_token_at: Optional[float] = None
                for response in mlx_stream_generate(
                    self._model,
                    self._tokenizer,
                    prompt=prompt,
                    **self._generation_kwargs(
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                    ),
                ):
                    usage = None
                    if first_token_at is None and getattr(response, "text", ""):
                        first_token_at = time.perf_counter()
                    if response.finish_reason is not None:
                        finished_at = time.perf_counter()
                        prompt_tokens = int(response.prompt_tokens)
                        completion_tokens = int(response.generation_tokens)
                        prompt_eval_duration = max((first_token_at or finished_at) - started_at, 0.0)
                        eval_duration = max(finished_at - (first_token_at or finished_at), 0.0)
                        usage = {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": int(response.prompt_tokens + response.generation_tokens),
                            "finish_reason": response.finish_reason,
                            "metrics": {
                                "total_duration_s": finished_at - started_at,
                                "prompt_eval_duration_s": prompt_eval_duration,
                                "prompt_eval_rate": (prompt_tokens / prompt_eval_duration) if prompt_eval_duration > 0 else None,
                                "eval_duration_s": eval_duration,
                                "eval_rate": (completion_tokens / eval_duration) if eval_duration > 0 else None,
                            },
                        }
                    yield response.text, usage
            except Exception as exc:
                raise InferenceError(str(exc)) from exc

    def chat(
        self,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> Tuple[str, dict]:
        """
        Run a chat completion using a list of ChatMessage objects.

        Applies the tokenizer's built-in chat template if available,
        otherwise falls back to a plain-text format.

        Args:
            messages: Conversation history (system + user + assistant turns).
            Other args: same as generate().

        Returns:
            Tuple of (assistant_response_text, usage_dict).

        Raises:
            InferenceError: if no model is loaded or generation fails.
        """
        with self._lock:
            self._require_loaded()

            prompt = self._apply_chat_template(messages)
            return self.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

    def chat_stream(
        self,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> Generator[Tuple[str, Optional[dict]], None, None]:
        """
        Stream chat completion chunks using the tokenizer's chat template.
        """
        with self._lock:
            self._require_loaded()

            prompt = self._apply_chat_template(messages)
            yield from self.generate_stream(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

    # ── Chat template helpers ────────────────────────────────────────────────

    def _generation_kwargs(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> dict:
        """Build shared MLX generation keyword arguments."""
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

    def _apply_chat_template(self, messages: List[ChatMessage]) -> str:
        """
        Convert a list of ChatMessage objects to a single prompt string.

        Preference order:
        1. tokenizer.apply_chat_template() — used if the tokenizer supports it.
        2. Simple fallback format.
        """
        tokenizer = self._tokenizer

        # Convert Pydantic models to dicts for the HF template API
        msg_dicts = [{"role": _openai_role_for_template(m.role), "content": m.content} for m in messages]

        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt: str = tokenizer.apply_chat_template(
                    msg_dicts,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return prompt
            except Exception as exc:
                logger.warning(
                    "apply_chat_template() failed (%s); using fallback format.", exc
                )

        return _fallback_chat_format(messages)


def _fallback_chat_format(messages: List[ChatMessage]) -> str:
    """
    Produce a plain-text prompt from chat messages when no chat template
    is available.

    Format:
        System: <system message>

        User: <user message>
        Assistant: <assistant message>
        User: <user message>
        Assistant:
    """
    parts: list[str] = []
    for msg in messages:
        if msg.role in {Role.system, Role.developer}:
            parts.append(f"System: {msg.content}\n")
        elif msg.role == Role.user:
            parts.append(f"User: {msg.content}")
        elif msg.role == Role.assistant:
            parts.append(f"Assistant: {msg.content}")
    parts.append("Assistant:")
    return "\n".join(parts)


def _openai_role_for_template(role: Role) -> str:
    """Map OpenAI roles onto the smaller role set most local chat templates expect."""
    if role == Role.developer:
        return Role.system.value
    return role.value
