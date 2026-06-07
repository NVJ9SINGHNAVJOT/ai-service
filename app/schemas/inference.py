"""
Pydantic schemas for inference endpoints.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Role(str, Enum):
    developer = "developer"
    system = "system"
    user = "user"
    assistant = "assistant"


class ChatMessage(BaseModel):
    """A single message in a conversation."""

    role: Role
    content: str | List[dict[str, Any]] | None = None

    def text_content(self) -> str:
        """Return only the textual portion of a possibly multimodal message."""
        if isinstance(self.content, str):
            return self.content
        if self.content is None:
            return ""

        text_parts: list[str] = []
        for item in self.content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in {"text", "input_text"}:
                text = item.get("text") or item.get("content") or ""
                if text:
                    text_parts.append(str(text))
        return " ".join(text_parts).strip()

    def image_inputs(self) -> List[str]:
        """Extract image references from OpenAI-style multimodal content parts."""
        if not isinstance(self.content, list):
            return []

        images: list[str] = []
        for item in self.content:
            if not isinstance(item, dict):
                continue

            item_type = item.get("type")
            if item_type == "image_url":
                payload = item.get("image_url")
                if isinstance(payload, dict):
                    url = payload.get("url")
                elif isinstance(payload, str):
                    url = payload
                else:
                    url = None
                if url:
                    images.append(str(url))
            elif item_type == "input_image":
                image_url = item.get("image_url")
                if image_url:
                    images.append(str(image_url))

        return images

    def audio_inputs(self) -> List[dict[str, Any]]:
        """Extract OpenAI-style ``input_audio`` payloads.

        Each payload is ``{"data": <base64-encoded audio>, "format": "wav"|...}``,
        matching what the OpenAI SDK sends. Decoding happens in the media service.
        """
        if not isinstance(self.content, list):
            return []

        audios: list[dict[str, Any]] = []
        for item in self.content:
            if not isinstance(item, dict):
                continue

            item_type = item.get("type")
            if item_type != "input_audio":
                continue

            payload = item.get("input_audio")
            if isinstance(payload, dict) and payload.get("data"):
                audios.append(payload)

        return audios

    def has_image(self) -> bool:
        """Return True when the message includes at least one image input."""
        return bool(self.image_inputs())

    def has_audio(self) -> bool:
        """Return True when the message includes at least one audio input."""
        return bool(self.audio_inputs())


class OpenAIChatCompletionRequest(BaseModel):
    """Subset of the OpenAI chat completions request supported by this server."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="Local model name")
    messages: List[ChatMessage] = Field(..., min_length=1, description="Conversation history")
    max_tokens: Optional[int] = Field(None, ge=1, le=32768)
    max_completion_tokens: Optional[int] = Field(None, ge=1, le=32768)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(None, ge=1.0, le=2.0)
    n: Optional[int] = Field(1, ge=1)
    stream: bool = Field(False, description="Whether to stream Server-Sent Events chunks")
    verbose: bool = Field(False, description="Whether to include server timing metrics in x_metrics")
    user: Optional[str] = Field(None, description="Opaque end-user identifier")
    metadata: Optional[dict[str, Any]] = None
    store: Optional[bool] = None
    service_tier: Optional[str] = None
    seed: Optional[int] = None
    safety_identifier: Optional[str] = None
    stream_options: Optional[dict[str, Any]] = None
    tools: Optional[Any] = None
    tool_choice: Optional[Any] = None
    parallel_tool_calls: Optional[bool] = None
    response_format: Optional[Any] = None
    function_call: Optional[Any] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[str | List[str]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    modalities: Optional[List[str]] = None
    audio: Optional[Any] = None
    prediction: Optional[Any] = None

    @model_validator(mode="after")
    def _normalize_token_limit(self) -> "OpenAIChatCompletionRequest":
        """Support OpenAI's max_completion_tokens alias."""
        if self.max_tokens is None and self.max_completion_tokens is not None:
            self.max_tokens = self.max_completion_tokens
        return self


class OpenAIChatCompletionMessage(BaseModel):
    """Assistant message shape returned inside an OpenAI-style choice."""

    role: str
    content: str


class OpenAIChatCompletionChoice(BaseModel):
    """Single completion choice in the OpenAI-compatible response."""

    index: int
    message: OpenAIChatCompletionMessage
    finish_reason: str


class OpenAIUsage(BaseModel):
    """Token usage summary expected by OpenAI-compatible clients."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIResponseMetrics(BaseModel):
    """Optional server-side timing metadata returned when verbose=true."""

    total_duration_s: Optional[float] = None
    load_duration_s: Optional[float] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration_s: Optional[float] = None
    prompt_eval_rate: Optional[float] = None
    eval_count: Optional[int] = None
    eval_duration_s: Optional[float] = None
    eval_rate: Optional[float] = None


class OpenAIChatCompletionResponse(BaseModel):
    """Top-level payload for `/v1/chat/completions`."""

    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAIChatCompletionChoice]
    usage: OpenAIUsage
    x_metrics: Optional[OpenAIResponseMetrics] = Field(
        None,
        description="Server-side timing metrics. `null` unless `verbose=true` was sent in the request.",
    )
