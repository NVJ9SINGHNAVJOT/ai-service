"""
Runtime monkeypatch for mlx-vlm's Gemma 4 loader — unused KV-shared weights.

Generic across all gemma4 models: it keys off each model's own
`num_kv_shared_layers`, so it fixes any KV-sharing repo (e.g. gemma-4-e4b) and is
a no-op for the rest (e.g. gemma-4-31b, gemma-4-26b-a4b).

This is a *workaround for a broken model upload*, deliberately kept out of the
main service code so the application logic stays clean —
`media_inference_service.py` only imports and calls
`patch_gemma4_shared_kv_load()`. See PATCHES.md (repo root) for the full
write-up, verification, and the condition under which this file can be deleted.

────────────────────────────────────────────────────────────────────────────
The bug (any KV-sharing Gemma 4 repo flagged `format: mlx`, on mlx-vlm 0.6.3)
────────────────────────────────────────────────────────────────────────────
Some Gemma 4 checkpoints use KV-sharing: a config with `num_kv_shared_layers: N`
(N > 0) means the last N layers reuse earlier layers' keys/values, so mlx-vlm
builds them with `kv_shared_only=True` and never creates their
`k_proj`/`v_proj`/`k_norm` modules — the checkpoint's copies are dead weight.
Concretely, `gemma-4-e4b-it` has 42 layers with 18 shared (indices 24–41), giving
18 × 3 = 54 such tensors. (Models with `num_kv_shared_layers: 0` — e.g.
`gemma-4-31b-it`, `gemma-4-26b-a4b-it` — have none, so the patch is a no-op there.)

mlx-vlm's own `LanguageModel.sanitize()` strips exactly these, but in
`mlx_vlm/utils.py::load_model` the entire sanitize block is gated on
`if not is_mlx_format:`. This repo's safetensors metadata is `{'format': 'mlx'}`,
so mlx-vlm treats it as already-sanitized, skips the strip, and the 54 redundant
tensors hit the strict `model.load_weights(...)`:

    ValueError: Received 54 parameters not in model:
    language_model.model.layers.24.self_attn.{k_norm,k_proj,v_proj}.weight  … 24–41

It is a packaging bug in the upload (flagged "MLX-format / no sanitization
needed" but the redundant weights were never stripped). Upgrading mlx/mlx-vlm
does not help — 0.6.3 is the latest and the defect is the model, not the library.

────────────────────────────────────────────────────────────────────────────
The fix
────────────────────────────────────────────────────────────────────────────
Override the gemma4 top-level `Model.load_weights` (inherited from
`mlx.nn.Module`) to drop the unused KV-shared tensors before delegating to the
original loader. We reuse mlx-vlm's own predicate
`LanguageModel._is_unused_shared_kv_weight` against the *built* model, so only
genuinely-shared layers are dropped — identical to what upstream's sanitize would
remove. The predicate reads each model's own `num_kv_shared_layers`, so this is
generic across all gemma4 repos and matches quantized weights too
(`.weight`/`.scales`/`.biases`). It is a no-op for non-KV-shared models (drops 0)
and for already-clean loads, and it does not change the forward pass: those layers
reuse KV from their source layers (whose projections load normally) and never call
`k_proj(x)`.

Idempotent and best-effort: a failure logs a warning and leaves loading otherwise
unchanged. On success you'll see this line the first time a media model loads:

    [INFO] app.patches.mlx_vlm_gemma4 — Applied mlx-vlm gemma4 KV-shared weight-strip patch.
"""

from __future__ import annotations

from app.core.logging import get_logger

logger = get_logger(__name__)


def patch_gemma4_shared_kv_load() -> None:
    """Strip unused KV-shared weights when loading Gemma 4 mlx-format repos (idempotent)."""
    try:
        import mlx.nn as nn
        from mlx_vlm.models.gemma4 import gemma4 as gemma4_mod

        model_cls = gemma4_mod.Model  # inherits load_weights from nn.Module
        if getattr(model_cls.load_weights, "_pp_gemma4_kv_patch", False):
            return

        original = nn.Module.load_weights

        def load_weights(self, weights, strict=True):
            # Drop the redundant copies of KV-shared layers' k/v projections that
            # this model class never allocates; upstream's sanitize is skipped for
            # `format: mlx` repos, so we replicate its drop here at load time.
            lm = getattr(self, "language_model", None)
            if (
                isinstance(weights, list)
                and lm is not None
                and hasattr(lm, "_is_unused_shared_kv_weight")
            ):
                weights = [
                    (k, v) for (k, v) in weights if not lm._is_unused_shared_kv_weight(k)
                ]
            return original(self, weights, strict=strict)

        load_weights._pp_gemma4_kv_patch = True
        model_cls.load_weights = load_weights
        logger.info("Applied mlx-vlm gemma4 KV-shared weight-strip patch.")
    except Exception as exc:  # noqa: BLE001 — patch is best-effort
        logger.warning("Could not apply gemma4 KV-shared weight-strip patch (%s).", exc)
