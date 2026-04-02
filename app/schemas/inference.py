"""
Pydantic schemas for inference endpoints.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


class ChatMessage(BaseModel):
    """A single message in a conversation."""

    role: Role
    content: str


class OpenAIChatCompletionRequest(BaseModel):
    """Subset of the OpenAI chat completions request supported by this server."""

    model: str = Field(..., description="Local model name")
    messages: List[ChatMessage] = Field(..., min_length=1, description="Conversation history")
    max_tokens: Optional[int] = Field(None, ge=1, le=32768)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(None, ge=1.0, le=2.0)
    stream: bool = Field(False, description="Whether to stream Server-Sent Events chunks")
    verbose: bool = Field(False, description="Whether to include server timing metrics in x_metrics")
    user: Optional[str] = Field(None, description="Opaque end-user identifier")


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
    x_metrics: Optional[OpenAIResponseMetrics] = None
