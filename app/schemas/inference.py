"""
Pydantic schemas for inference endpoints.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


class ChatMessage(BaseModel):
    """A single message in a conversation."""

    role: Role
    content: str


class GenerateRequest(BaseModel):
    """Request body for POST /api/v1/inference/generate."""

    model: str = Field(..., description="Local (sanitised) model name to use for inference")
    prompt: str = Field(..., description="Text prompt to complete")
    system_prompt: Optional[str] = Field(
        None,
        description="Optional system-level instruction prepended to the prompt",
    )
    max_tokens: Optional[int] = Field(None, ge=1, le=32768, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(None, ge=1.0, le=2.0)
    stream: bool = Field(False, description="Whether to stream incremental output chunks")


class ChatRequest(BaseModel):
    """Request body for POST /api/v1/inference/chat (conversation-style)."""

    model: str = Field(..., description="Local (sanitised) model name")
    messages: List[ChatMessage] = Field(..., min_length=1, description="Conversation history")
    max_tokens: Optional[int] = Field(None, ge=1, le=32768)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(None, ge=1.0, le=2.0)
    stream: bool = Field(False, description="Whether to stream incremental output chunks")


class GenerateResponse(BaseModel):
    """Response from /generate or /chat."""

    success: bool
    message: str
    data: Optional[Any] = None


class GenerateData(BaseModel):
    """Payload inside GenerateResponse.data."""

    model: str
    text: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class OpenAIChatCompletionRequest(BaseModel):
    """Subset of the OpenAI chat completions request supported by this server."""

    model: str = Field(..., description="Local model name")
    messages: List[ChatMessage] = Field(..., min_length=1, description="Conversation history")
    max_tokens: Optional[int] = Field(None, ge=1, le=32768)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    stream: bool = Field(False, description="Whether to stream Server-Sent Events chunks")
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


class OpenAIChatCompletionResponse(BaseModel):
    """Top-level payload for `/v1/chat/completions`."""

    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAIChatCompletionChoice]
    usage: OpenAIUsage
