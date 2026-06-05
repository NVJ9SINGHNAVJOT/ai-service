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

import time
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple

from app.config import Settings
from app.core.exceptions import InferenceError, ModelLoadError
from app.core.logging import get_logger
from app.schemas.inference import ChatMessage, Role
from app.services.base_inference_service import LoadedModelService, openai_role_for_template

logger = get_logger(__name__)


class InferenceService(LoadedModelService):
    """
    Manages a single loaded MLX-LM (text) model and exposes inference methods.

    Lifecycle bookkeeping (lock, loaded-model state, runtime marker) lives in
    :class:`LoadedModelService`; this subclass owns the mlx-lm specifics.
    """

    def __init__(self, cfg: Optional[Settings] = None) -> None:
        super().__init__(cfg=cfg)
        self._tokenizer: Optional[Any] = None

    # ── Load / unload ────────────────────────────────────────────────────────

    def _release_backend(self) -> None:
        """Drop the tokenizer handle on unload (model reference cleared by base)."""
        self._tokenizer = None

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
                last_response: Optional[Any] = None
                generated_parts: list[str] = []
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
                    last_response = response
                    text = getattr(response, "text", "") or ""
                    if text:
                        generated_parts.append(text)
                    usage = None
                    if first_token_at is None and text:
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
                    yield text, usage
                if last_response is not None and getattr(last_response, "finish_reason", None) is None:
                    finished_at = time.perf_counter()
                    prompt_eval_duration = max((first_token_at or finished_at) - started_at, 0.0)
                    eval_duration = max(finished_at - (first_token_at or finished_at), 0.0)
                    prompt_tokens = self._count_tokens(prompt)
                    completion_tokens = self._count_tokens("".join(generated_parts))
                    usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": (
                            prompt_tokens + completion_tokens
                            if prompt_tokens is not None and completion_tokens is not None
                            else None
                        ),
                        "finish_reason": "stop",
                        "metrics": {
                            "total_duration_s": finished_at - started_at,
                            "prompt_eval_duration_s": prompt_eval_duration,
                            "prompt_eval_rate": (
                                prompt_tokens / prompt_eval_duration
                                if prompt_tokens is not None and prompt_eval_duration > 0
                                else None
                            ),
                            "eval_duration_s": eval_duration,
                            "eval_rate": (
                                completion_tokens / eval_duration
                                if completion_tokens is not None and eval_duration > 0
                                else None
                            ),
                        },
                    }
                    yield "", usage
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
        chunks: list[str] = []
        usage: dict = {}
        for chunk, chunk_usage in self.chat_stream(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        ):
            chunks.append(chunk)
            if chunk_usage is not None:
                usage = chunk_usage
        return "".join(chunks), usage

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
            stop_markers = _chat_stop_markers(self._tokenizer)
            if not stop_markers:
                yield from self.generate_stream(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
                return

            raw_stream = self.generate_stream(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            max_marker_len = max(len(item) for item in stop_markers)
            pending = ""
            visible_parts: list[str] = []
            started_at = time.perf_counter()
            first_visible_at: Optional[float] = None
            final_usage: Optional[dict] = None

            for chunk, usage in raw_stream:
                pending += chunk
                trimmed, found_stop = _split_at_first_stop_marker(pending, stop_markers)
                if found_stop:
                    if trimmed:
                        if first_visible_at is None:
                            first_visible_at = time.perf_counter()
                        visible_parts.append(trimmed)
                        yield trimmed, None

                    if usage is None:
                        finished_at = time.perf_counter()
                        prompt_eval_duration = max((first_visible_at or finished_at) - started_at, 0.0)
                        eval_duration = max(finished_at - (first_visible_at or finished_at), 0.0)
                        prompt_tokens = self._count_tokens(prompt)
                        completion_tokens = self._count_tokens("".join(visible_parts))
                        final_usage = {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": (
                                prompt_tokens + completion_tokens
                                if prompt_tokens is not None and completion_tokens is not None
                                else None
                            ),
                            "finish_reason": "stop",
                            "metrics": {
                                "total_duration_s": finished_at - started_at,
                                "prompt_eval_duration_s": prompt_eval_duration,
                                "prompt_eval_rate": (
                                    prompt_tokens / prompt_eval_duration
                                    if prompt_tokens is not None and prompt_eval_duration > 0
                                    else None
                                ),
                                "eval_duration_s": eval_duration,
                                "eval_rate": (
                                    completion_tokens / eval_duration
                                    if completion_tokens is not None and eval_duration > 0
                                    else None
                                ),
                            },
                        }
                    else:
                        final_usage = {**usage, "finish_reason": "stop"}

                    yield "", final_usage
                    return

                safe_len = max(0, len(pending) - (max_marker_len - 1))
                if safe_len > 0:
                    safe_text = pending[:safe_len]
                    pending = pending[safe_len:]
                    if safe_text:
                        if first_visible_at is None:
                            first_visible_at = time.perf_counter()
                        visible_parts.append(safe_text)
                        yield safe_text, None
                if usage is not None:
                    final_usage = usage

            if pending:
                if first_visible_at is None and pending:
                    first_visible_at = time.perf_counter()
                visible_parts.append(pending)
                yield pending, None

            if final_usage is not None:
                yield "", final_usage

    # ── Chat template helpers ────────────────────────────────────────────────

    def _apply_chat_template(self, messages: List[ChatMessage]) -> str:
        """
        Convert a list of ChatMessage objects to a single prompt string.

        Preference order:
        1. tokenizer.apply_chat_template() — used if the tokenizer supports it.
        2. Simple fallback format.
        """
        tokenizer = self._tokenizer

        # Convert Pydantic models to dicts for the HF template API
        msg_dicts = [{"role": openai_role_for_template(m.role), "content": m.content} for m in messages]

        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt: str = tokenizer.apply_chat_template(
                    msg_dicts,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return prompt
            except Exception as exc:
                user_first_messages = _messages_for_user_first_template(messages)
                if user_first_messages is not None:
                    try:
                        prompt = tokenizer.apply_chat_template(
                            user_first_messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        logger.info(
                            "apply_chat_template() required a user-first conversation; folded system instructions into the first user turn."
                        )
                        return prompt
                    except Exception:
                        pass
                logger.warning(
                    "apply_chat_template() failed (%s); using fallback format.", exc
                )

        return _fallback_chat_format(messages)

    def _count_tokens(self, text: str) -> Optional[int]:
        """Best-effort token counting for verbose fallbacks."""
        tokenizer = self._tokenizer
        if tokenizer is None:
            return None

        try:
            if hasattr(tokenizer, "encode"):
                tokens = tokenizer.encode(text)
                return len(tokens)
        except Exception:
            return None
        return None


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


def _messages_for_user_first_template(messages: List[ChatMessage]) -> Optional[list[dict[str, str]]]:
    """
    Fold leading system/developer guidance into the first user turn.

    Some tokenizer templates reject conversations that begin with `system`.
    For those models we preserve the instruction by prepending it to the first
    user message instead of abandoning the tokenizer's native template.
    """
    system_parts: list[str] = []
    rebuilt: list[dict[str, str]] = []
    first_user_index: Optional[int] = None

    for msg in messages:
        if first_user_index is None and msg.role in {Role.system, Role.developer}:
            if msg.content:
                system_parts.append(str(msg.content))
            continue

        content = msg.content if isinstance(msg.content, str) else msg.text_content()
        rebuilt.append({"role": openai_role_for_template(msg.role), "content": content or ""})
        if first_user_index is None and msg.role == Role.user:
            first_user_index = len(rebuilt) - 1

    if first_user_index is None:
        return None

    if system_parts:
        merged_instruction = "\n\n".join(system_parts).strip()
        first_user = rebuilt[first_user_index]
        existing = first_user.get("content", "")
        first_user["content"] = (
            f"{merged_instruction}\n\n{existing}".strip() if existing else merged_instruction
        )

    return rebuilt


def _chat_stop_markers(tokenizer: Any) -> list[str]:
    """Return known special markers that should not leak into chat output."""
    markers = {
        "<end_of_turn>",
        "<|eot_id|>",
        "<|end_of_text|>",
    }

    for attr in ("eos_token", "sep_token"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, str) and value.startswith("<") and value.endswith(">"):
            markers.add(value)

    additional = getattr(tokenizer, "additional_special_tokens", None)
    if isinstance(additional, list):
        for item in additional:
            if isinstance(item, str) and item.startswith("<") and item.endswith(">"):
                markers.add(item)

    return sorted(marker for marker in markers if marker)


def _split_at_first_stop_marker(text: str, stop_markers: list[str]) -> tuple[str, bool]:
    """Return text trimmed at the earliest matching stop marker."""
    first_index: Optional[int] = None
    for marker in stop_markers:
        index = text.find(marker)
        if index >= 0 and (first_index is None or index < first_index):
            first_index = index

    if first_index is None:
        return text, False
    return text[:first_index], True
