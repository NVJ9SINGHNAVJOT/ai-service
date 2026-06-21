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
import ast
import importlib.util
import re
import shutil
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

from app.config import Settings, settings as _default_settings
from app.core.exceptions import (
    DownloadError,
    InvalidModelPathError,
    ModelAlreadyExistsError,
    ModelBusyError,
    ModelNotFoundError,
    RegistryError,
    UnsupportedModelError,
)
from app.core.logging import get_logger
from app.schemas.model import ModelInfo, ModelSource, ModelState
from app.services.model_runtime_state import ModelRuntimeState, RuntimeModelActivity

logger = get_logger(__name__)

# Files whose presence indicates a directory is likely a valid MLX/HF model.
_MODEL_INDICATOR_FILES = {"config.json", "tokenizer_config.json"}


@dataclass
class ModelDiagnosis:
    """Human-oriented troubleshooting snapshot for a local model."""

    name: str
    source: ModelSource
    state: ModelState
    loadable: bool
    path: str
    repo_id: Optional[str]
    model_type: Optional[str]
    effective_model_type: Optional[str]
    backend: str
    supported_by_mlx: Optional[bool]
    input_modalities: List[str]
    missing_files: List[str]
    summary: str
    recommendations: List[str]


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


def _read_model_type(path: Path) -> Optional[str]:
    """Best-effort read of the model_type from config.json."""
    config_path = path / "config.json"
    if not config_path.exists():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    model_type = data.get("model_type")
    return model_type if isinstance(model_type, str) and model_type else None


def _read_model_config(path: Path) -> dict:
    """Best-effort read of config.json for diagnosis helpers."""
    config_path = path / "config.json"
    if not config_path.exists():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_max_context_tokens(path: Path) -> Optional[int]:
    """Best-effort read of a model's context window from config.json.

    Tries the common keys in order, including the nested ``text_config`` block
    used by multimodal models. Returns ``None`` when nothing usable is found.
    """
    config = _read_model_config(path)
    text_config = config.get("text_config")
    candidates = [
        config.get("max_position_embeddings"),
        text_config.get("max_position_embeddings") if isinstance(text_config, dict) else None,
        config.get("n_positions"),
        config.get("n_ctx"),
        config.get("max_sequence_length"),
    ]
    for value in candidates:
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            return value
    return None


def _detect_input_modalities(path: Path) -> List[str]:
    """
    Infer supported input types from the model config.

    This is a best-effort heuristic used for doctor output. It intentionally
    favors being helpful over being overly strict.
    """
    config = _read_model_config(path)
    model_type = str(config.get("model_type") or "").lower()
    modalities = ["text"]

    # vision_config must be a non-empty dict — a null/empty entry means no vision encoder
    has_vision_config = isinstance(config.get("vision_config"), dict) and bool(config["vision_config"])
    vision_keys = {
        "vision_tower",
        "vision_encoder",
        "mm_vision_tower",
    }
    vision_hints = ("vl", "vision", "llava", "paligemma", "mllama", "gemma4", "pixtral", "ocr")
    if has_vision_config or any(key in config for key in vision_keys) or any(hint in model_type for hint in vision_hints):
        modalities.append("image")

    audio_keys = {
        "audio_config",
        "audio_token_index",
        "speech_config",
        "audio_encoder",
        "num_audio_tokens",
    }
    audio_hints = ("audio", "omni", "speech", "voice")
    if any(key in config for key in audio_keys) or any(hint in model_type for hint in audio_hints):
        modalities.append("audio")

    video_keys = {
        "video_token_index",
        "num_video_tokens",
        "video_encoder",
        "video_config",
    }
    video_hints = ("video", "vid2seq", "videollm")
    if any(key in config for key in video_keys) or any(hint in model_type for hint in video_hints):
        modalities.append("video")

    return modalities


@lru_cache(maxsize=1)
def _supported_mlx_model_types() -> Optional[Set[str]]:
    """Return model backend names supported by the installed mlx_lm package."""
    spec = importlib.util.find_spec("mlx_lm")
    if spec is None or not spec.submodule_search_locations:
        return None

    root = Path(next(iter(spec.submodule_search_locations)))
    models_dir = root / "models"
    if not models_dir.exists():
        return None

    return {
        path.stem
        for path in models_dir.glob("*.py")
        if path.stem != "__init__"
    }


@lru_cache(maxsize=1)
def _mlx_model_remapping() -> Dict[str, str]:
    """Parse MODEL_REMAPPING from the installed mlx_lm utils file without importing MLX."""
    spec = importlib.util.find_spec("mlx_lm")
    if spec is None or not spec.submodule_search_locations:
        return {}

    root = Path(next(iter(spec.submodule_search_locations)))
    utils_path = root / "utils.py"
    if not utils_path.exists():
        return {}

    try:
        tree = ast.parse(utils_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return {}

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MODEL_REMAPPING":
                    try:
                        value = ast.literal_eval(node.value)
                    except (ValueError, SyntaxError):
                        return {}
                    if isinstance(value, dict):
                        return {
                            str(key): str(mapped)
                            for key, mapped in value.items()
                        }
    return {}


@lru_cache(maxsize=1)
def _supported_mlx_vlm_model_types() -> Optional[Set[str]]:
    """Return model type names supported by the installed mlx_vlm package.

    Includes both directory names under mlx_vlm/models/ and all keys in
    mlx_vlm's MODEL_REMAPPING (which are aliases that also route through vlm).
    Returns None when mlx_vlm is not installed.
    """
    spec = importlib.util.find_spec("mlx_vlm")
    if spec is None or not spec.submodule_search_locations:
        return None

    root = Path(next(iter(spec.submodule_search_locations)))
    models_dir = root / "models"
    if not models_dir.exists():
        return None

    types: Set[str] = set()

    # Architecture directories
    for path in models_dir.iterdir():
        if path.is_dir() and not path.name.startswith("_"):
            types.add(path.name)
        elif path.suffix == ".py" and path.stem != "__init__":
            types.add(path.stem)

    # MODEL_REMAPPING aliases
    utils_path = root / "utils.py"
    if utils_path.exists():
        try:
            tree = ast.parse(utils_path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            tree = None
        if tree:
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "MODEL_REMAPPING":
                            try:
                                value = ast.literal_eval(node.value)
                            except (ValueError, SyntaxError):
                                break
                            if isinstance(value, dict):
                                types.update(str(k) for k in value)

    return types or None


def _detect_backend(path: Path) -> str:
    """Determine the correct inference backend based on the model's conversion tool.

    Three-way registry check (no direct dependency on modalities):

    1. model_type only in mlx-vlm registry → mlx-vlm  (exclusively a VLM architecture)
    2. model_type in both registries       → mlx-vlm only if vision_config is a
                                             non-empty dict (the model was packed as
                                             a full VLM, not just the text backbone)
    3. model_type only in mlx-lm or unknown → mlx-lm

    _detect_input_modalities() is kept entirely separate — it describes what inputs
    the model accepts and is never called here.
    """
    model_type = (_read_model_type(path) or "").lower()
    if not model_type:
        return "mlx-lm"

    vlm_types = _supported_mlx_vlm_model_types()
    lm_types = _supported_mlx_model_types()

    in_vlm = vlm_types is not None and model_type in vlm_types
    in_lm = lm_types is not None and (
        model_type in lm_types or _mlx_model_remapping().get(model_type, model_type) in lm_types
    )

    if not in_vlm:
        return "mlx-lm"

    if not in_lm:
        # Exclusively in mlx-vlm — must use vlm loader
        return "mlx-vlm"

    # In both registries: the architecture can be text-only (mlx-lm backbone) or
    # full VLM. Distinguish by checking vision_config in config.json directly —
    # a non-empty dict means the model was exported as a full VLM.
    config = _read_model_config(path)
    if isinstance(config.get("vision_config"), dict) and config["vision_config"]:
        return "mlx-vlm"

    return "mlx-lm"


def _is_model_type_supported(path: Path) -> bool:
    """
    Return True when the model architecture appears supported by installed mlx_lm.

    If support metadata cannot be determined, we return True to avoid false
    negatives. The runtime loader remains the final source of truth.
    """
    supported = _supported_mlx_model_types()
    if supported is None:
        return True

    model_type = _read_model_type(path)
    if not model_type:
        return True

    effective_type = _mlx_model_remapping().get(model_type, model_type)
    return effective_type in supported


def _is_model_type_supported_for_backend(path: Path, backend: str) -> bool:
    """
    Return True when the architecture is supported by the backend that will
    actually load it.

    Loading routes purely on ``backend`` (see ``_detect_backend``), so a VLM-only
    architecture must be validated against mlx_vlm, not mlx_lm. Checking it against
    mlx_lm would flag every multimodal model as unsupported even though the vlm
    loader handles it fine.
    """
    if backend != "mlx-vlm":
        return _is_model_type_supported(path)

    supported = _supported_mlx_vlm_model_types()
    if supported is None:
        return True

    model_type = (_read_model_type(path) or "").lower()
    if not model_type:
        return True
    return model_type in supported


def _effective_model_type(model_type: Optional[str]) -> Optional[str]:
    """Return the effective mlx_lm backend name after remapping."""
    if not model_type:
        return None
    return _mlx_model_remapping().get(model_type, model_type)


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
        self._runtime_state = ModelRuntimeState(cfg=self._cfg)

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
        runtime_activity: Optional[RuntimeModelActivity] = None,
    ) -> ModelInfo:
        """
        Build a ``ModelInfo`` instance from a model directory on disk.

        Downloaded models may have extra metadata in the registry JSON, while
        custom models are filesystem-only. Centralizing this mapping makes the
        list output easier to keep consistent.
        """
        reg = registry_entry or {}
        raw_loadable = _is_loadable(path)
        backend = _detect_backend(path)
        model_type_supported = _is_model_type_supported_for_backend(path, backend)
        state = _resolve_model_state(raw_loadable, model_type_supported, runtime_activity)
        input_modalities = _detect_input_modalities(path)
        return ModelInfo(
            name=path.name,
            repo_id=reg.get("repo_id") or (runtime_activity.repo_id if runtime_activity else None),
            source=source,
            state=state,
            path=str(path.resolve()),
            loadable=_resolve_loadable(raw_loadable, model_type_supported, state),
            input_modalities=input_modalities,
            backend=backend,
            max_context_tokens=_read_max_context_tokens(path),
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
        runtime_activity = self._runtime_state.snapshot()
        models: List[ModelInfo] = []
        seen_names: set[str] = set()

        # Downloaded models
        dl_dir = self._cfg.downloaded_models_path
        if dl_dir.exists():
            for entry in sorted(dl_dir.iterdir()):
                if not entry.is_dir():
                    continue
                seen_names.add(entry.name)
                models.append(
                    self._build_model_info(
                        path=entry,
                        source=ModelSource.downloaded,
                        registry_entry=registry.get(entry.name, {}),
                        runtime_activity=runtime_activity.get(entry.name),
                    )
                )

        # Custom models
        custom_dir = self._cfg.custom_models_path
        if custom_dir.exists():
            for entry in sorted(custom_dir.iterdir()):
                if not entry.is_dir():
                    continue
                seen_names.add(entry.name)
                models.append(
                    self._build_model_info(
                        path=entry,
                        source=ModelSource.custom,
                        runtime_activity=runtime_activity.get(entry.name),
                    )
                )

        for name, activity in sorted(runtime_activity.items()):
            if name in seen_names:
                continue
            models.append(
                self._build_model_info(
                    path=self._safe_downloaded_path(name),
                    source=ModelSource.downloaded,
                    registry_entry=registry.get(name, {}),
                    runtime_activity=activity,
                )
            )

        models.sort(key=lambda model: model.name.lower())
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

    def diagnose_model(self, name: str) -> ModelDiagnosis:
        """Return a troubleshooting snapshot for one local model."""
        info = self.get_model(name)
        return self._diagnose_model_info(info)

    def diagnose_models(self) -> List[ModelDiagnosis]:
        """Return troubleshooting snapshots for all known local models."""
        return [self._diagnose_model_info(info) for info in self.list_models()]

    def _ensure_model_not_busy(self, name: str) -> None:
        """Block mutating operations for models that are running or downloading."""
        info = self.get_model(name)
        if info.state in {ModelState.running, ModelState.downloading}:
            raise ModelBusyError(name, info.state.value)

    def ensure_model_loadable(self, name: str) -> ModelInfo:
        """Return model info or raise a clear error if the model is not loadable."""
        info = self.get_model(name)
        model_path = Path(info.path)

        if not _is_loadable(model_path):
            raise InvalidModelPathError(
                f"Model '{name}' is missing expected files like config.json or tokenizer_config.json."
            )

        model_type = _read_model_type(model_path)
        if model_type and not _is_model_type_supported(model_path):
            raise UnsupportedModelError(name, model_type)

        logger.info("Loading '%s' via mlx-lm (model_type: %s, modalities: %s)", name, model_type or "unknown", ", ".join(info.input_modalities))
        return info

    def ensure_model_files_ready(self, name: str) -> ModelInfo:
        """
        Return model info when the on-disk files look complete enough to use.

        This check intentionally does not enforce mlx-lm architecture support,
        because a model may be unsupported by mlx-lm but still usable through
        mlx-vlm or another runtime.
        """
        info = self.get_model(name)
        model_path = Path(info.path)
        if not _is_loadable(model_path):
            raise InvalidModelPathError(
                f"Model '{name}' is missing expected files like config.json or tokenizer_config.json."
            )
        model_type = _read_model_type(model_path)
        logger.info("Loading '%s' via mlx-vlm (model_type: %s, modalities: %s)", name, model_type or "unknown", ", ".join(info.input_modalities))
        return info

    def _diagnose_model_info(self, info: ModelInfo) -> ModelDiagnosis:
        """Compute a CLI-friendly diagnosis for a model."""
        model_path = Path(info.path)
        backend = info.backend
        model_type = _read_model_type(model_path)
        effective_model_type = _effective_model_type(model_type)
        supported_by_mlx = (
            None if not model_type else _is_model_type_supported_for_backend(model_path, backend)
        )
        input_modalities = _detect_input_modalities(model_path)
        missing_files = [
            filename for filename in sorted(_MODEL_INDICATOR_FILES) if not (model_path / filename).exists()
        ]

        recommendations: List[str] = []
        if info.state == ModelState.downloading:
            summary = "Download is still in progress."
            recommendations.append("Wait for the download to finish, then run the doctor command again.")
        elif info.state == ModelState.running:
            summary = "Model is currently in use by another process."
            recommendations.append("Stop the active chat or server process before modifying the model.")
        elif missing_files:
            summary = "Model directory is incomplete."
            recommendations.append("Re-download the model or restore the missing files.")
        elif model_type and supported_by_mlx is False:
            package = "mlx-vlm" if backend == "mlx-vlm" else "mlx-lm"
            summary = f"Installed {package} does not support model_type '{model_type}'."
            recommendations.append(f"Upgrade `mlx` and `{package}`, or choose a model with a supported model_type.")
        elif info.loadable:
            summary = "Model looks ready to load."
            recommendations.append("If loading still fails, run the raw `mlx_lm.generate` command to isolate runtime issues.")
        else:
            summary = "Model is not loadable."
            recommendations.append("Check that the directory contains a complete Hugging Face / MLX model layout.")

        return ModelDiagnosis(
            name=info.name,
            source=info.source,
            state=info.state,
            loadable=info.loadable,
            path=info.path,
            repo_id=info.repo_id,
            model_type=model_type,
            effective_model_type=effective_model_type,
            backend=backend,
            supported_by_mlx=supported_by_mlx,
            input_modalities=input_modalities,
            missing_files=missing_files,
            summary=summary,
            recommendations=recommendations,
        )

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
        name = _sanitize_repo_id(repo_id)
        dest = self._safe_downloaded_path(name)

        if dest.exists() and not force:
            raise ModelAlreadyExistsError(name)

        if dest.exists() and force:
            logger.info("Removing existing model directory for forced re-download: %s", dest)
            shutil.rmtree(dest)

        self._download_snapshot(repo_id, name, dest)

        self._register(name, repo_id, dest)
        logger.info("Model '%s' downloaded successfully.", name)
        return self.get_model(name)

    def update(self, name: str) -> ModelInfo:
        """
        Update a downloaded model using a staged re-download.

        Requires the model to be in the registry (needs original repo_id).

        Args:
            name: Local (sanitised) model name.

        Returns:
            Updated ModelInfo.

        Raises:
            ModelNotFoundError: if model is not found or not in registry.
            DownloadError: if re-download fails.
        """
        self._ensure_model_not_busy(name)
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
        logger.info("Updating model '%s' (repo: %s) via staged re-download", name, repo_id)

        dest = self._safe_downloaded_path(name)
        updates_dir = self._cfg.runtime_path / "updates"
        updates_dir.mkdir(parents=True, exist_ok=True)
        staged_dest = Path(tempfile.mkdtemp(prefix=f"{name}.", dir=str(updates_dir)))
        backup_dest: Optional[Path] = None

        try:
            self._download_snapshot(repo_id, name, staged_dest)

            if dest.exists():
                backup_dest = Path(tempfile.mkdtemp(prefix=f".{name}.backup.", dir=str(dest.parent)))
                shutil.rmtree(backup_dest)
                shutil.move(str(dest), str(backup_dest))

            shutil.move(str(staged_dest), str(dest))
            self._register(name, repo_id, dest)

        except BaseException:
            if backup_dest and backup_dest.exists() and not dest.exists():
                shutil.move(str(backup_dest), str(dest))
            raise
        finally:
            if staged_dest.exists():
                shutil.rmtree(staged_dest, ignore_errors=True)
            if backup_dest and backup_dest.exists():
                shutil.rmtree(backup_dest, ignore_errors=True)

        logger.info("Model '%s' updated successfully.", name)
        return self.get_model(name)

    def _download_snapshot(self, repo_id: str, name: str, dest: Path) -> None:
        """Download a repository snapshot into ``dest`` and clean partial files."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise DownloadError(repo_id, "huggingface_hub is not installed") from exc

        logger.info("Downloading model '%s' → %s", repo_id, dest)
        download_marker = self._runtime_state.mark_downloading(name, repo_id)

        kwargs: dict = {
            "repo_id": repo_id,
            "local_dir": str(dest),
            "local_dir_use_symlinks": False,
        }
        if self._cfg.hf_token:
            kwargs["token"] = self._cfg.hf_token

        try:
            snapshot_download(**kwargs)
        except KeyboardInterrupt as exc:
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            raise DownloadError(repo_id, "download interrupted by user") from exc
        except Exception as exc:
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            raise DownloadError(repo_id, str(exc)) from exc
        finally:
            self._runtime_state.clear_marker(download_marker)

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
        self._ensure_model_not_busy(name)
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


def _resolve_model_state(
    loadable: bool,
    model_type_supported: bool,
    runtime_activity: Optional[RuntimeModelActivity],
) -> ModelState:
    """Derive the user-facing state from live activity and on-disk validity."""
    if runtime_activity and runtime_activity.downloading:
        return ModelState.downloading
    if runtime_activity and runtime_activity.running:
        return ModelState.running
    if loadable and not model_type_supported:
        return ModelState.unsupported
    if loadable:
        return ModelState.ready
    return ModelState.incomplete


def _resolve_loadable(loadable: bool, model_type_supported: bool, state: ModelState) -> bool:
    """Hide transiently incomplete models from the loadable column."""
    if state == ModelState.downloading:
        return False
    return loadable and model_type_supported
