"""
ModelManager service.

Responsibilities:
- Maintain a JSON registry of downloaded models.
- Download models from HuggingFace using huggingface_hub.snapshot_download().
- List models from both the downloaded/ and custom/ directories.
- Update (delete + re-download) and delete models safely.
- Prevent directory-traversal attacks.

Download strategy rationale
────────────────────────────
MLX-LM's mlx_lm.load() expects a local directory containing the standard
HuggingFace file layout: config.json, *.safetensors (or *.npz for MLX
native), tokenizer files, etc.

huggingface_hub.snapshot_download() is the most reliable way to mirror
an entire HF repo to disk while respecting caching, partial downloads,
and auth tokens. The result is a flat directory that mlx_lm.load() can
read directly.

We deliberately avoid shelling out to `mlx_lm.convert` here because:
  1. MLX-community repos are already in MLX format — no conversion needed.
  2. Conversion doubles disk space and adds complexity.
  3. Users who need to convert PyTorch models can do so separately.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app.config import Settings, settings as _default_settings
from app.core.exceptions import (
    DownloadError,
    InvalidModelPathError,
    ModelAlreadyExistsError,
    ModelNotFoundError,
    RegistryError,
)
from app.core.logging import get_logger
from app.schemas.model import ModelInfo, ModelSource

logger = get_logger(__name__)

# Files whose presence indicates a directory is likely a valid MLX/HF model.
_MODEL_INDICATOR_FILES = {"config.json", "tokenizer_config.json"}


def _sanitize_repo_id(repo_id: str) -> str:
    """
    Convert a HuggingFace repo ID to a safe local folder name.

    Examples:
        mlx-community/Llama-3.2-3B-Instruct-4bit  →  mlx-community__Llama-3.2-3B-Instruct-4bit
        org/model name with spaces                 →  org__model-name-with-spaces
    """
    # Replace path separator first
    name = repo_id.replace("/", "__")
    # Replace any remaining characters that are unsafe in directory names
    name = re.sub(r"[^\w\-.]", "-", name)
    return name


def _dir_size_mb(path: Path) -> float:
    """Return the total size of a directory in megabytes (best-effort)."""
    try:
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return round(total / (1024 * 1024), 2)
    except Exception:
        return 0.0


def _is_loadable(path: Path) -> bool:
    """Return True if the directory looks like a valid model (has key config files)."""
    if not path.is_dir():
        return False
    files = {f.name for f in path.iterdir()}
    return bool(_MODEL_INDICATOR_FILES & files)


def _is_within_base(path: Path, base: Path) -> bool:
    """
    Return True when ``path`` resolves inside ``base`` (or is exactly ``base``).

    This helper keeps our directory traversal checks consistent anywhere we need
    to confirm that a user-provided model path stays inside an allowed folder.
    """
    path = path.resolve()
    base = base.resolve()
    return path == base or str(path).startswith(str(base) + "/")


class ModelManager:
    """
    Manages the lifecycle of local MLX-LM models.

    This is a stateless service (no in-memory model objects); it works
    exclusively with the filesystem and the registry JSON.
    """

    def __init__(self, cfg: Optional[Settings] = None) -> None:
        self._cfg = cfg or _default_settings
        self._cfg.ensure_directories()

    # ── Registry helpers ─────────────────────────────────────────────────────

    def _load_registry(self) -> Dict[str, dict]:
        """Load and return the registry dict.  Returns {} if not found."""
        path = self._cfg.registry_path
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise RegistryError(f"Failed to read registry at {path}: {exc}") from exc

    def _save_registry(self, registry: Dict[str, dict]) -> None:
        """Persist the registry dict to disk."""
        path = self._cfg.registry_path
        try:
            path.write_text(
                json.dumps(registry, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as exc:
            raise RegistryError(f"Failed to write registry at {path}: {exc}") from exc

    def _register(self, name: str, repo_id: str, path: Path) -> None:
        """Add or update a registry entry for a downloaded model."""
        registry = self._load_registry()
        now = datetime.now(timezone.utc).isoformat()
        existing = registry.get(name, {})
        registry[name] = {
            "name": name,
            "repo_id": repo_id,
            "path": str(path),
            "source": ModelSource.downloaded.value,
            "created_at": existing.get("created_at", now),
            "updated_at": now,
        }
        self._save_registry(registry)

    def _unregister(self, name: str) -> None:
        """Remove a registry entry."""
        registry = self._load_registry()
        registry.pop(name, None)
        self._save_registry(registry)

    # ── Security ─────────────────────────────────────────────────────────────

    def _safe_downloaded_path(self, name: str) -> Path:
        """
        Return the expected path for a downloaded model and verify it stays
        within the downloaded_models directory (prevents directory traversal).
        """
        base = self._cfg.downloaded_models_path.resolve()
        candidate = (base / name).resolve()
        if not _is_within_base(candidate, base):
            raise InvalidModelPathError(
                f"'{name}' resolves outside the allowed models directory."
            )
        return candidate

    def _build_model_info(
        self,
        path: Path,
        source: ModelSource,
        registry_entry: Optional[dict] = None,
    ) -> ModelInfo:
        """
        Build a ``ModelInfo`` instance from a model directory on disk.

        Downloaded models may have extra metadata in the registry JSON, while
        custom models are filesystem-only. Centralizing this mapping makes the
        list output easier to keep consistent.
        """
        reg = registry_entry or {}
        return ModelInfo(
            name=path.name,
            repo_id=reg.get("repo_id"),
            source=source,
            path=str(path.resolve()),
            loadable=_is_loadable(path),
            size_mb=_dir_size_mb(path),
            created_at=_parse_dt(reg.get("created_at")),
            updated_at=_parse_dt(reg.get("updated_at")),
        )

    # ── Public API ───────────────────────────────────────────────────────────

    def list_models(self) -> List[ModelInfo]:
        """
        Return metadata for all models in downloaded/ and custom/ directories.
        Registry information is merged in for downloaded models.
        """
        registry = self._load_registry()
        models: List[ModelInfo] = []

        # Downloaded models
        dl_dir = self._cfg.downloaded_models_path
        if dl_dir.exists():
            for entry in sorted(dl_dir.iterdir()):
                if not entry.is_dir():
                    continue
                models.append(
                    self._build_model_info(
                        path=entry,
                        source=ModelSource.downloaded,
                        registry_entry=registry.get(entry.name, {}),
                    )
                )

        # Custom models
        custom_dir = self._cfg.custom_models_path
        if custom_dir.exists():
            for entry in sorted(custom_dir.iterdir()):
                if not entry.is_dir():
                    continue
                models.append(
                    self._build_model_info(
                        path=entry,
                        source=ModelSource.custom,
                    )
                )

        return models

    def get_model(self, name: str) -> ModelInfo:
        """
        Look up a single model by its local name.
        Searches downloaded/ first, then custom/.
        """
        for m in self.list_models():
            if m.name == name:
                return m
        raise ModelNotFoundError(name)

    def get_model_path(self, name: str) -> Path:
        """Return the filesystem Path for the named model (must exist)."""
        info = self.get_model(name)
        return Path(info.path)

    def download(self, repo_id: str, force: bool = False) -> ModelInfo:
        """
        Download an MLX-compatible HuggingFace model.

        Uses huggingface_hub.snapshot_download() to mirror the entire
        repository to models/downloaded/<sanitized-name>/.

        Args:
            repo_id: HuggingFace repository ID, e.g. 'mlx-community/Llama-3.2-3B-Instruct-4bit'
            force:   If True, overwrite any existing download.

        Returns:
            ModelInfo for the newly downloaded model.

        Raises:
            ModelAlreadyExistsError: if model exists and force=False.
            DownloadError: if the download fails.
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise DownloadError(repo_id, "huggingface_hub is not installed") from exc

        name = _sanitize_repo_id(repo_id)
        dest = self._safe_downloaded_path(name)

        if dest.exists() and not force:
            raise ModelAlreadyExistsError(name)

        if dest.exists() and force:
            logger.info("Removing existing model directory for forced re-download: %s", dest)
            shutil.rmtree(dest)

        logger.info("Downloading model '%s' → %s", repo_id, dest)

        kwargs: dict = {
            "repo_id": repo_id,
            "local_dir": str(dest),
            "local_dir_use_symlinks": False,
        }
        if self._cfg.hf_token:
            kwargs["token"] = self._cfg.hf_token

        try:
            snapshot_download(**kwargs)
        except Exception as exc:
            # Clean up a partially downloaded directory
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            raise DownloadError(repo_id, str(exc)) from exc

        self._register(name, repo_id, dest)
        logger.info("Model '%s' downloaded successfully.", name)
        return self.get_model(name)

    def update(self, name: str) -> ModelInfo:
        """
        Update a downloaded model by deleting and re-downloading it.

        Requires the model to be in the registry (needs original repo_id).

        Args:
            name: Local (sanitised) model name.

        Returns:
            Updated ModelInfo.

        Raises:
            ModelNotFoundError: if model is not found or not in registry.
            DownloadError: if re-download fails.
        """
        registry = self._load_registry()
        entry = registry.get(name)
        if entry is None:
            # Maybe it exists on disk but isn't in the registry
            info = self.get_model(name)  # raises ModelNotFoundError if truly absent
            if info.source == ModelSource.custom:
                raise InvalidModelPathError(
                    "Cannot update a custom model via the update command. "
                    "Replace the files manually in models/custom/."
                )
            raise ModelNotFoundError(
                f"Model '{name}' exists on disk but is not in the registry. "
                "Cannot determine original repo_id for re-download."
            )

        repo_id = entry["repo_id"]
        logger.info("Updating model '%s' (repo: %s) — delete + re-download", name, repo_id)

        # Delete existing files
        dest = self._safe_downloaded_path(name)
        if dest.exists():
            shutil.rmtree(dest)
        self._unregister(name)

        # Re-download (force=True because we just deleted it)
        return self.download(repo_id, force=True)

    def delete(self, name: str, allow_custom: bool = False) -> None:
        """
        Delete a local model.

        By default only downloaded models can be deleted through this method.
        Pass allow_custom=True to also permit deletion of custom models.

        Args:
            name:         Local (sanitised) model name.
            allow_custom: If True, custom models can also be deleted.

        Raises:
            ModelNotFoundError:    if the model does not exist.
            InvalidModelPathError: if attempting to delete a custom model without permission,
                                   or if the path is outside allowed directories.
        """
        info = self.get_model(name)

        if info.source == ModelSource.custom and not allow_custom:
            raise InvalidModelPathError(
                f"'{name}' is a custom model. "
                "Pass --allow-custom / allow_custom=True to delete it."
            )

        target = Path(info.path).resolve()

        # Safety check: ensure we are deleting within an allowed directory
        allowed_bases = [
            self._cfg.downloaded_models_path.resolve(),
            self._cfg.custom_models_path.resolve(),
        ]
        if not any(_is_within_base(target, base) for base in allowed_bases):
            raise InvalidModelPathError(
                f"Refusing to delete '{target}' — it is outside the allowed model directories."
            )

        logger.info("Deleting model '%s' at %s", name, target)
        shutil.rmtree(target)

        if info.source == ModelSource.downloaded:
            self._unregister(name)

        logger.info("Model '%s' deleted.", name)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-format datetime string, returning None on failure."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
