"""
Runtime monkeypatch for mlx-audio's Kokoro TTS resampling.

This is a *workaround for a third-party bug*, deliberately kept out of the main
service code so the application logic stays clean — `audio_service.py` only
imports and calls `patch_interpolate_ceil_drift()`. See PATCHES.md (repo root)
for the full write-up, verification, and the condition under which this file can
be deleted.

────────────────────────────────────────────────────────────────────────────
The bug (mlx-audio 0.4.4)
────────────────────────────────────────────────────────────────────────────
`mlx_audio.tts.models.interpolate.interpolate()` derives its output length with
`math.ceil(in_len * scale_factor)`. Commit aaf5ee6 ("Fix Kokoro usage from
worker threads") swapped `mx.ceil` for Python's `math.ceil`, which runs on
float64. Because `1/300` is not exactly representable in binary floating point,
some lengths land just above an integer:

    296400 * (1/300) == 296400 * 0.0033333333333333335 == 988.0000000000001

`mx.ceil` (float32) truncated that to 988.0; `math.ceil` rounds the float64
value up to 989. Kokoro's `SineGen._f02sine` downsamples by `1/upsample_scale`
then upsamples back by `upsample_scale`, so the off-by-one compounds:

    down: ceil(296400 * 1/300) = 989      (should be 988)
    up:   ceil(989   * 300)    = 296700   (should be 296400)

`sine_waves` (resampled → 296700) and `uv` (direct from f0 → 296400) are then
combined with a broadcast that fails:

    ValueError: [broadcast_shapes] Shapes (1,296400,1) and (1,296700,9) cannot be broadcast

Short inputs stay aligned, which is why only longer replies crash.

────────────────────────────────────────────────────────────────────────────
The fix
────────────────────────────────────────────────────────────────────────────
Wrap `interpolate` so the length is computed with a tiny epsilon tolerance
before the ceiling (988.0000000000001 → 988.0), then delegate to the original
implementation with an explicit `size`. The buggy code only runs on the
`scale_factor` path, so the `size` path is passed straight through.

`kokoro/istftnet.py` did `from ..interpolate import interpolate`, so it holds
its own reference to the function — the patch rebinds *both* the source module
attribute and the istftnet binding. Idempotent and best-effort: a failure logs a
warning and leaves TTS otherwise running.
"""

from __future__ import annotations

from app.core.logging import get_logger

logger = get_logger(__name__)


def patch_interpolate_ceil_drift() -> None:
    """Apply the mlx-audio interpolate ceil-drift workaround (idempotent)."""
    try:
        import math

        from mlx_audio.tts.models import interpolate as interp_mod
        from mlx_audio.tts.models.kokoro import istftnet

        if getattr(interp_mod.interpolate, "_pp_ceil_patch", False):
            return

        original = interp_mod.interpolate

        def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
            # Only the scale_factor path carries the float-ceil drift; compute
            # size with an epsilon tolerance and delegate via explicit size.
            if size is None and scale_factor is not None:
                spatial = input.ndim - 2
                factors = (
                    scale_factor
                    if isinstance(scale_factor, (list, tuple))
                    else [scale_factor] * spatial
                )
                size = []
                for i in range(spatial):
                    val = float(input.shape[i + 2]) * float(factors[i])
                    rounded = round(val, 9)
                    if abs(val - rounded) < 1e-9:  # 988.0000000000001 -> 988.0
                        val = rounded
                    size.append(max(1, int(math.ceil(val))))
                return original(input, size=size, mode=mode, align_corners=align_corners)
            return original(
                input, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners
            )

        interpolate._pp_ceil_patch = True
        interp_mod.interpolate = interpolate
        istftnet.interpolate = interpolate  # rebind the name already imported into istftnet
        logger.info("Applied mlx-audio interpolate ceil-drift patch.")
    except Exception as exc:  # noqa: BLE001 — patch is best-effort
        logger.warning("Could not apply interpolate ceil-drift patch (%s).", exc)
