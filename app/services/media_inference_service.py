"""
MediaInferenceService — loads an mlx-vlm model and serves multimodal chat.

This mirrors the text-only InferenceService closely so the OpenAI-compatible
API can switch between `mlx_lm` and `mlx_vlm` based on whether a request
includes image or audio content.
"""

from __future__ import annotations

import base64
import binascii
import importlib
import importlib.util
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple

from app.config import Settings
from app.core.exceptions import InferenceError, ModelLoadError
from app.core.logging import get_logger
from app.patches import patch_gemma4_shared_kv_load
from app.schemas.inference import ChatMessage, Role
from app.services.base_inference_service import LoadedModelService, openai_role_for_template

logger = get_logger(__name__)


def _strip_audio_data_uri(data: str) -> str:
    """Accept a bare base64 string or a ``data:audio/...;base64,<payload>`` URI."""
    if data.startswith("data:") and ";base64," in data:
        return data.split(";base64,", 1)[1]
    return data


def _write_audio_temp(payload: dict[str, Any]) -> str:
    """Decode one OpenAI ``input_audio`` payload to a temp file and return its path."""
    fmt = str(payload.get("format") or "wav").lower().lstrip(".")
    suffix = "." + (fmt if fmt.isalnum() else "wav")
    cleaned = "".join(_strip_audio_data_uri(str(payload.get("data") or "")).split())
    try:
        raw = base64.b64decode(cleaned, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InferenceError(
            "input_audio.data must be base64-encoded audio bytes (as the OpenAI SDK sends)."
        ) from exc
    if not raw:
        raise InferenceError("input_audio.data was empty after base64 decoding.")

    fd, path = tempfile.mkstemp(suffix=suffix, prefix="ai_audio_")
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)
    return path


@contextmanager
def _decode_audio_to_files(payloads: List[dict[str, Any]]) -> Generator[List[str], None, None]:
    """Materialize base64 ``input_audio`` payloads as temp files for mlx-vlm.

    mlx-vlm reads audio from a file path/URL, not raw base64, so we decode each
    payload to a short-lived temp file and clean them all up once generation ends.
    """
    temp_paths: list[str] = []
    try:
        for payload in payloads:
            temp_paths.append(_write_audio_temp(payload))
        yield temp_paths
    finally:
        for path in temp_paths:
            try:
                os.unlink(path)
            except OSError:
                pass


class MediaInferenceService(LoadedModelService):
    """
    Manages a single loaded mlx-vlm (multimodal) model for API inference.

    Lifecycle bookkeeping (lock, loaded-model state, runtime marker) lives in
    :class:`LoadedModelService`; this subclass owns the mlx-vlm specifics.
    """

    _NOT_LOADED_MESSAGE = "No media model is currently loaded. Call load() first."

    def __init__(self, cfg: Optional[Settings] = None) -> None:
        super().__init__(cfg=cfg)
        self._processor: Optional[Any] = None
        self._stream_generate: Optional[Any] = None
        self._apply_chat_template: Optional[Any] = None

    def _release_backend(self) -> None:
        """Drop mlx-vlm handles on unload (model reference cleared by base)."""
        self._processor = None
        self._stream_generate = None
        self._apply_chat_template = None

    def load(self, model_path: Path, model_name: str) -> None:
        """Load an mlx-vlm model from disk."""
        with self._lock:
            if self._loaded_name == model_name:
                if self._running_marker is None:
                    self._running_marker = self._runtime_state.mark_running(model_name)
                logger.debug("Media model '%s' is already loaded.", model_name)
                return

            if self._model is not None:
                logger.info("Unloading current media model '%s' to load '%s'.", self._loaded_name, model_name)
                self._unload_internal()

            logger.info("Loading media model '%s' from %s …", model_name, model_path)
            try:
                if importlib.util.find_spec("mlx_vlm") is None:
                    raise RuntimeError("`mlx-vlm` is not installed.")

                mlx_vlm = importlib.import_module("mlx_vlm")
                generate_mod = importlib.import_module("mlx_vlm.generate")
                prompt_utils_mod = importlib.import_module("mlx_vlm.prompt_utils")

                patch_gemma4_shared_kv_load()

                started_at = time.perf_counter()
                self._model, self._processor = mlx_vlm.load(str(model_path))
                self._stream_generate = generate_mod.stream_generate
                self._apply_chat_template = prompt_utils_mod.apply_chat_template
                self._last_load_duration_s = time.perf_counter() - started_at
                self._loaded_name = model_name
                self._running_marker = self._runtime_state.mark_running(model_name)
                logger.info("Media model '%s' loaded successfully.", model_name)
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
            images, audio_payloads = self._extract_media(messages)
            with _decode_audio_to_files(audio_payloads) as audios:
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

    @staticmethod
    def _prepare_messages(messages: List[ChatMessage]) -> List[dict]:
        """Convert request messages into the text-only payload expected by the template."""
        return [
            {
                "role": openai_role_for_template(message.role),
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
