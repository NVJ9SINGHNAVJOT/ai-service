"""
Lightweight cross-process runtime state for model activity.

The CLI can be invoked from multiple terminals, so transient states like
"downloading" or "running" cannot live only in memory. We persist those
states as tiny marker files under ``models/runtime/`` and prune stale markers
when the owning PID is no longer alive.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from app.config import Settings, settings as _default_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RuntimeModelActivity:
    """Aggregate live activity we discovered for a model across processes."""

    downloading: int = 0
    running: int = 0
    repo_id: Optional[str] = None


class ModelRuntimeState:
    """Persist transient model lifecycle state in process-owned marker files."""

    def __init__(self, cfg: Optional[Settings] = None) -> None:
        self._cfg = cfg or _default_settings
        self._cfg.ensure_directories()

    @property
    def _downloads_dir(self) -> Path:
        return self._cfg.runtime_path / "downloads"

    @property
    def _usage_dir(self) -> Path:
        return self._cfg.runtime_path / "usage"

    def ensure_directories(self) -> None:
        """Create the runtime marker layout if it does not exist yet."""
        self._downloads_dir.mkdir(parents=True, exist_ok=True)
        self._usage_dir.mkdir(parents=True, exist_ok=True)

    def mark_downloading(self, model_name: str, repo_id: Optional[str] = None) -> Path:
        """Create and return a download marker owned by this process."""
        return self._write_marker(
            directory=self._downloads_dir,
            activity="downloading",
            model_name=model_name,
            repo_id=repo_id,
        )

    def mark_running(self, model_name: str) -> Path:
        """Create and return a running marker owned by this process."""
        return self._write_marker(
            directory=self._usage_dir,
            activity="running",
            model_name=model_name,
        )

    def clear_marker(self, marker_path: Optional[Path]) -> None:
        """Remove a marker if it still exists."""
        if marker_path is None:
            return
        try:
            marker_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.debug("Failed to remove runtime marker %s: %s", marker_path, exc)

    def snapshot(self) -> Dict[str, RuntimeModelActivity]:
        """
        Return aggregated runtime activity keyed by model name.

        Stale markers are removed opportunistically when their PID is no longer
        alive.
        """
        self.ensure_directories()
        models: Dict[str, RuntimeModelActivity] = {}

        for directory, field_name in (
            (self._downloads_dir, "downloading"),
            (self._usage_dir, "running"),
        ):
            for marker_path in directory.glob("*.json"):
                payload = self._read_marker(marker_path)
                if payload is None:
                    continue

                model_name = payload.get("model_name")
                if not isinstance(model_name, str) or not model_name:
                    self.clear_marker(marker_path)
                    continue

                activity = models.setdefault(model_name, RuntimeModelActivity())
                setattr(activity, field_name, getattr(activity, field_name) + 1)

                repo_id = payload.get("repo_id")
                if isinstance(repo_id, str) and repo_id and not activity.repo_id:
                    activity.repo_id = repo_id

        return models

    def _write_marker(
        self,
        directory: Path,
        activity: str,
        model_name: str,
        repo_id: Optional[str] = None,
    ) -> Path:
        self.ensure_directories()
        marker_path = directory / f"{os.getpid()}-{time.time_ns()}.json"
        payload = {
            "activity": activity,
            "model_name": model_name,
            "repo_id": repo_id,
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        marker_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return marker_path

    def _read_marker(self, marker_path: Path) -> Optional[dict]:
        try:
            payload = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.clear_marker(marker_path)
            return None

        pid = payload.get("pid")
        if not isinstance(pid, int) or not _is_pid_alive(pid):
            self.clear_marker(marker_path)
            return None
        return payload


def _is_pid_alive(pid: int) -> bool:
    """Return True when the process still exists on this machine."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True
