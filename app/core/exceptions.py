"""
Domain exceptions for the AI Service.

All custom exceptions inherit from `MLXManagerError` so callers can catch
the entire domain with a single except clause if desired.
"""

from __future__ import annotations


class MLXManagerError(Exception):
    """Base exception for all AI Service domain errors."""


class ModelNotFoundError(MLXManagerError):
    """Raised when a requested model does not exist locally."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Model not found: '{name}'")
        self.name = name


class ModelAlreadyExistsError(MLXManagerError):
    """Raised when trying to download a model that already exists (and not forcing)."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Model already exists: '{name}'. Use update to refresh it.")
        self.name = name


class InvalidModelPathError(MLXManagerError):
    """Raised when a model path fails security or validity checks."""

    def __init__(self, detail: str) -> None:
        super().__init__(f"Invalid model path: {detail}")


class ModelLoadError(MLXManagerError):
    """Raised when a model cannot be loaded into memory for inference."""

    def __init__(self, name: str, reason: str) -> None:
        super().__init__(f"Failed to load model '{name}': {reason}")
        self.name = name
        self.reason = reason


class InferenceError(MLXManagerError):
    """Raised when an inference call fails."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Inference error: {reason}")
        self.reason = reason


class DownloadError(MLXManagerError):
    """Raised when a model download fails."""

    def __init__(self, repo_id: str, reason: str) -> None:
        super().__init__(f"Download failed for '{repo_id}': {reason}")
        self.repo_id = repo_id
        self.reason = reason


class RegistryError(MLXManagerError):
    """Raised on registry read/write failures."""


class ModelBusyError(MLXManagerError):
    """Raised when an operation is attempted on a model that is currently active."""

    def __init__(self, name: str, state: str) -> None:
        super().__init__(f"Model '{name}' is currently {state} and cannot be modified right now.")
        self.name = name
        self.state = state


class UnsupportedModelError(MLXManagerError):
    """Raised when the local MLX installation does not support a model architecture."""

    def __init__(self, name: str, model_type: str) -> None:
        super().__init__(
            f"Model '{name}' is not supported by the installed mlx_lm runtime "
            f"(unsupported model_type: '{model_type}')."
        )
        self.name = name
        self.model_type = model_type


class MediaChatError(MLXManagerError):
    """Raised when a media chat invocation cannot be started or fails."""

    def __init__(self, detail: str) -> None:
        super().__init__(f"Media chat failed: {detail}")
