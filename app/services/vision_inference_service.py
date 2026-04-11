"""
VisionInferenceService — loads an mlx-vlm model and serves multimodal chat.

This mirrors the text-only InferenceService closely so the OpenAI-compatible
API can switch between `mlx_lm` and `mlx_vlm` based on whether a request
includes image or audio content.
"""

from __future__ import annotations

import importlib
import importlib.util
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


class VisionInferenceService:
    """Manages a single loaded mlx-vlm model for API inference."""

    def __init__(self, cfg: Optional[Settings] = None) -> None:
        self._cfg = cfg or _default_settings
        self._lock = threading.RLock()
        self._loaded_name: Optional[str] = None
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None
        self._stream_generate: Optional[Any] = None
        self._apply_chat_template: Optional[Any] = None
        self._last_load_duration_s: Optional[float] = None
        self._runtime_state = ModelRuntimeState(cfg=self._cfg)
        self._running_marker: Optional[Path] = None

    @property
    def loaded_model_name(self) -> Optional[str]:
        """Name of the currently loaded vision model, or None."""
        return self._loaded_name

    @property
    def is_loaded(self) -> bool:
        """Return True when a vision model is resident in memory."""
        return self._model is not None

    @property
    def last_load_duration_s(self) -> Optional[float]:
        """Duration of the most recent successful model load, in seconds."""
        return self._last_load_duration_s

    def load(self, model_path: Path, model_name: str) -> None:
        """Load an mlx-vlm model from disk."""
        with self._lock:
            if self._loaded_name == model_name:
                if self._running_marker is None:
                    self._running_marker = self._runtime_state.mark_running(model_name)
                logger.debug("Vision model '%s' is already loaded.", model_name)
                return

            if self._model is not None:
                logger.info("Unloading current vision model '%s' to load '%s'.", self._loaded_name, model_name)
                self._unload_internal()

            logger.info("Loading vision model '%s' from %s …", model_name, model_path)
            try:
                if importlib.util.find_spec("mlx_vlm") is None:
                    raise RuntimeError("`mlx-vlm` is not installed.")

                mlx_vlm = importlib.import_module("mlx_vlm")
                generate_mod = importlib.import_module("mlx_vlm.generate")
                prompt_utils_mod = importlib.import_module("mlx_vlm.prompt_utils")

                started_at = time.perf_counter()
                self._model, self._processor = mlx_vlm.load(str(model_path))
                self._stream_generate = generate_mod.stream_generate
                self._apply_chat_template = prompt_utils_mod.apply_chat_template
                self._last_load_duration_s = time.perf_counter() - started_at
                self._loaded_name = model_name
                self._running_marker = self._runtime_state.mark_running(model_name)
                logger.info("Vision model '%s' loaded successfully.", model_name)
            except Exception as exc:
                self._model = None
                self._processor = None
                self._stream_generate = None
                self._apply_chat_template = None
                self._loaded_name = None
                self._last_load_duration_s = None
                self._runtime_state.clear_marker(self._running_marker)
                self._running_marker = None
                raise ModelLoadError(model_name, str(exc)) from exc

    def unload(self) -> Optional[str]:
        """Unload the currently loaded vision model."""
        with self._lock:
            name = self._loaded_name
            self._unload_internal()
            return name

    def _require_loaded(self) -> None:
        if not self.is_loaded:
            raise InferenceError("No vision model is currently loaded. Call load() first.")

    def _unload_internal(self) -> None:
        self._runtime_state.clear_marker(self._running_marker)
        self._running_marker = None
        self._model = None
        self._processor = None
        self._stream_generate = None
        self._apply_chat_template = None
        self._loaded_name = None
        self._last_load_duration_s = None

    def chat(
        self,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> Tuple[str, dict]:
        """Run a multimodal chat completion and return the buffered text."""
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
        """Stream one multimodal chat completion."""
        with self._lock:
            self._require_loaded()

            processed_messages = self._prepare_messages(messages)
            images, audios = self._extract_media(messages)
            prompt = self._apply_chat_template(
                self._processor,
                self._model.config,
                processed_messages,
                num_images=len(images),
                num_audios=len(audios),
            )

            try:
                started_at = time.perf_counter()
                first_token_at: Optional[float] = None

                for response in self._stream_generate(
                    model=self._model,
                    processor=self._processor,
                    prompt=prompt,
                    image=images or None,
                    audio=audios or None,
                    **self._generation_kwargs(
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                    ),
                ):
                    text = getattr(response, "text", "") or ""
                    usage = None
                    if first_token_at is None and text:
                        first_token_at = time.perf_counter()
                    if getattr(response, "finish_reason", None) is not None:
                        finished_at = time.perf_counter()
                        prompt_tokens = int(getattr(response, "prompt_tokens", 0) or 0)
                        completion_tokens = int(getattr(response, "generation_tokens", 0) or 0)
                        prompt_eval_duration = max((first_token_at or finished_at) - started_at, 0.0)
                        eval_duration = max(finished_at - (first_token_at or finished_at), 0.0)
                        usage = {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                            "finish_reason": getattr(response, "finish_reason", "stop"),
                            "metrics": {
                                "total_duration_s": finished_at - started_at,
                                "prompt_eval_duration_s": prompt_eval_duration,
                                "prompt_eval_rate": (prompt_tokens / prompt_eval_duration) if prompt_eval_duration > 0 else None,
                                "eval_duration_s": eval_duration,
                                "eval_rate": (completion_tokens / eval_duration) if eval_duration > 0 else None,
                            },
                        }
                    yield text, usage
            except Exception as exc:
                raise InferenceError(str(exc)) from exc

    def _generation_kwargs(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> dict:
        """Build shared generation keyword arguments for mlx-vlm."""
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

    @staticmethod
    def _prepare_messages(messages: List[ChatMessage]) -> List[dict]:
        """Convert request messages into the text-only payload expected by the template."""
        return [
            {
                "role": _openai_role_for_template(message.role),
                "content": message.text_content(),
            }
            for message in messages
        ]

    @staticmethod
    def _extract_media(messages: List[ChatMessage]) -> tuple[List[str], List[str]]:
        """
        Extract media inputs from the latest user turns that include them.

        The API should use the most recent user-provided images and audios
        present in the request body for the current assistant turn.
        """
        images: list[str] = []
        audios: list[str] = []
        for message in reversed(messages):
            if message.role != Role.user:
                continue
            if not images:
                images = message.image_inputs()
            if not audios:
                audios = message.audio_inputs()
            if images and audios:
                break
        return images, audios


def _openai_role_for_template(role: Role) -> str:
    """Map OpenAI roles onto the smaller role set most local chat templates expect."""
    if role == Role.developer:
        return Role.system.value
    return role.value
