"""
Application configuration.

Settings are loaded from environment variables (or a .env file).
All path settings resolve relative to the project root if not absolute.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root is two levels up from this file (app/config.py → app/ → project root)
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or `.env`.

    Path fields are kept as strings here because that maps cleanly to env vars;
    the resolved ``Path`` objects are exposed through convenience properties
    below so the rest of the app can work with absolute paths safely.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Model storage ───────────────────────────────────────────────────────
    models_base_dir: str = "models"
    downloaded_models_dir: str = "models/downloaded"
    custom_models_dir: str = "models/custom"
    model_registry_file: str = "models/registry.json"
    model_runtime_dir: str = "models/runtime"

    # ── API server ──────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Inference defaults ──────────────────────────────────────────────────
    default_max_tokens: int = 512
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    default_repetition_penalty: float = 1.1

    # ── HuggingFace ─────────────────────────────────────────────────────────
    hf_token: Optional[str] = None

    # ── Resolved paths (computed properties) ───────────────────────────────

    @property
    def downloaded_models_path(self) -> Path:
        """Resolved absolute path to the downloaded models directory."""
        return _resolve_path(self.downloaded_models_dir)

    @property
    def custom_models_path(self) -> Path:
        """Resolved absolute path to the custom models directory."""
        return _resolve_path(self.custom_models_dir)

    @property
    def registry_path(self) -> Path:
        """Resolved absolute path to the registry JSON file."""
        return _resolve_path(self.model_registry_file)

    @property
    def runtime_path(self) -> Path:
        """Resolved absolute path to the runtime state directory."""
        return _resolve_path(self.model_runtime_dir)

    def ensure_directories(self) -> None:
        """
        Create the filesystem layout expected by the app.

        This is safe to call repeatedly and is used by both the API server and
        services that work directly with the models directory.
        """
        self.downloaded_models_path.mkdir(parents=True, exist_ok=True)
        self.custom_models_path.mkdir(parents=True, exist_ok=True)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.runtime_path.mkdir(parents=True, exist_ok=True)


def _resolve_path(path_str: str) -> Path:
    """Return an absolute Path, resolving relative paths against the project root."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (_PROJECT_ROOT / p).resolve()


# Module-level singleton — import this in all other modules.
settings = Settings()
