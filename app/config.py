"""
Application configuration.

Settings are loaded from environment variables (or a .env file).
All path settings resolve relative to the project root if not absolute.
"""

from __future__ import annotations

import os
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

    # ── Swagger / OpenAPI examples ──────────────────────────────────────────
    # Model names pre-filled into the Swagger "Try it out" example bodies
    # (chat completions + model load/unload). Set these once here (or in .env)
    # to the models you have locally; the examples read from them at startup.
    # Changing a value requires a server restart to take effect.
    example_text_model: str = "mlx-community__Meta-Llama-3.1-8B-Instruct-8bit"
    example_media_model: str = "mlx-community__gemma-4-e4b-it-bf16"

    # ── HuggingFace ─────────────────────────────────────────────────────────
    hf_token: Optional[str] = None
    # Cache for HuggingFace model weights (Whisper/Kokoro, etc.). Kept inside the
    # project by default so downloads don't land in the global ~/.cache/huggingface.
    hf_cache_dir: str = "models/hf-cache"

    # ── Audio (STT / TTS) ───────────────────────────────────────────────────
    # Downloaded to the HuggingFace cache on first use. Whisper turbo gives the
    # best accuracy/speed for feeding an LLM; swap to a lighter repo (e.g.
    # "mlx-community/whisper-base-mlx") to cut the download and memory footprint.
    stt_model: str = "mlx-community/whisper-large-v3-turbo"
    tts_model: str = "prince-canuma/Kokoro-82M"
    tts_voice: str = "af_heart"
    tts_lang_code: str = "a"  # Kokoro: 'a' = American English, 'b' = British

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

    @property
    def hf_cache_path(self) -> Path:
        """Resolved absolute path to the project-local HuggingFace cache."""
        return _resolve_path(self.hf_cache_dir)

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
        self.hf_cache_path.mkdir(parents=True, exist_ok=True)


def _resolve_path(path_str: str) -> Path:
    """Return an absolute Path, resolving relative paths against the project root."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (_PROJECT_ROOT / p).resolve()


# Module-level singleton — import this in all other modules.
settings = Settings()

# Route HuggingFace downloads (Whisper/Kokoro weights, etc.) into the project's
# models/ dir instead of the global ~/.cache/huggingface. Must run before
# huggingface_hub is first imported anywhere; an explicit HF_HUB_CACHE still wins.
os.environ.setdefault("HF_HUB_CACHE", str(settings.hf_cache_path))
