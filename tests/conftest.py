from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def tmp_models_dir(tmp_path: Path):
    """Provide a temporary models directory tree."""
    downloaded = tmp_path / "downloaded"
    custom = tmp_path / "custom"
    downloaded.mkdir()
    custom.mkdir()
    return tmp_path


@pytest.fixture()
def manager(tmp_models_dir: Path):
    """ModelManager pointing at a temporary directory."""
    from app.config import Settings
    from app.services.model_manager import ModelManager

    cfg = Settings(
        downloaded_models_dir=str(tmp_models_dir / "downloaded"),
        custom_models_dir=str(tmp_models_dir / "custom"),
        model_registry_file=str(tmp_models_dir / "registry.json"),
        model_runtime_dir=str(tmp_models_dir / "runtime"),
    )
    return ModelManager(cfg=cfg)


@pytest.fixture()
def api_client():
    """TestClient for the FastAPI application."""
    from app.main import app

    return TestClient(app)
