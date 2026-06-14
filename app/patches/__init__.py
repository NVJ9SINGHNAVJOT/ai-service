"""
Runtime monkeypatches for third-party dependencies.

Each patch lives in its own fully-commented module here so the application code
only ever imports and calls it — the workarounds stay isolated from the main
logic. See PATCHES.md (repo root) for the rationale, verification, and removal
conditions of each patch.
"""

from app.patches.mlx_audio_kokoro import patch_interpolate_ceil_drift

__all__ = ["patch_interpolate_ceil_drift"]
