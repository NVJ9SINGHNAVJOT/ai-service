"""
Pydantic schemas for model management endpoints.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ModelSource(str, Enum):
    """Where the model files came from."""

    downloaded = "downloaded"
    custom = "custom"


class ModelState(str, Enum):
    """Current local lifecycle state of a model."""

    ready = "ready"
    downloading = "downloading"
    running = "running"
    incomplete = "incomplete"


class ModelInfo(BaseModel):
    """Metadata about a single locally available model."""

    name: str = Field(..., description="Sanitised local folder name used as identifier")
    repo_id: Optional[str] = Field(None, description="Original HuggingFace repo ID, if known")
    source: ModelSource = Field(..., description="Whether the model was downloaded or added manually")
    state: ModelState = Field(ModelState.ready, description="Current local lifecycle state")
    path: str = Field(..., description="Absolute path to the model directory")
    loadable: bool = Field(..., description="True if the directory contains the expected model files")
    size_mb: Optional[float] = Field(None, description="Approximate total size in MB")
    created_at: Optional[datetime] = Field(None, description="When the model was first registered")
    updated_at: Optional[datetime] = Field(None, description="When the model was last updated")


class LoadModelRequest(BaseModel):
    """Request body for POST /api/v1/models/load."""

    name: str = Field(..., description="Local (sanitised) model name to load into memory")


class UnloadModelRequest(BaseModel):
    """Request body for POST /api/v1/models/unload."""

    name: Optional[str] = Field(
        None,
        description="Name of the model to unload. If omitted, unloads whatever is currently loaded.",
    )


# ── Generic API wrapper ──────────────────────────────────────────────────────

class APIResponse(BaseModel):
    """Standard envelope for all API responses."""

    success: bool
    message: str
    data: Optional[Any] = None
