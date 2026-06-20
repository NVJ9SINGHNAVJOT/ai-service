# PATCHES

This file documents **runtime monkeypatches** this service applies to its
third-party dependencies. These are *not* changes to the libraries on disk —
they are applied in our own code at startup so they survive `pip install` /
reinstalls and travel with the repo. Each entry should say what is patched, why,
how to verify it, and the condition under which it can be removed.

---

## 1. mlx-audio — Kokoro TTS resample length drift (`interpolate` ceil)

| | |
|---|---|
| **Dependency** | `mlx-audio==0.4.4` (latest at time of writing) |
| **Where applied** | `app/patches/mlx_audio_kokoro.py` → `patch_interpolate_ceil_drift()`, called from `app/services/audio_service.py::_ensure_tts_loaded()` before the model loads |
| **Patched symbol** | `mlx_audio.tts.models.interpolate.interpolate` (and the rebound copy in `mlx_audio.tts.models.kokoro.istftnet`) |
| **Status** | Stopgap — remove once upstream ships the fix (see *Removal*) |

### Symptom

`POST /v1/audio/speech` returns **500** on longer replies (short ones work). The
server log shows:

```text
ValueError: [broadcast_shapes] Shapes (1,296400,1) and (1,296700,9) cannot be broadcast.
```

…raised deep inside Kokoro's iSTFTNet source module
(`istftnet.py` → `SineGen.__call__` → `_f02sine`).

### Root cause

`interpolate()` derives its output length with
`math.ceil(in_len * scale_factor)`. In commit `aaf5ee6` ("Fix Kokoro usage from
worker threads") upstream replaced `mx.ceil` with Python's `math.ceil` to avoid
compilation locks on worker threads.

`math.ceil` operates on **float64**, and `1/300` is not exactly representable in
binary floating point, so for some lengths the product lands just above an
integer:

```text
296400 * (1/300) == 296400 * 0.0033333333333333335 == 988.0000000000001
```

`mx.ceil` (on float32) truncated this back to `988.0`; `math.ceil` rounds the
float64 value up to **989**. `SineGen._f02sine` downsamples by `1/upsample_scale`
then upsamples by `upsample_scale`, so this off-by-one compounds:

```text
down: ceil(296400 * 1/300) = 989   (should be 988)
up:   ceil(989   * 300)     = 296700  (should be 296400)
```

The `sine_waves` (resampled, now 296700) and `uv` (computed directly from `f0`,
still 296400) are then combined with a broadcast that fails. Whether it triggers
depends on the exact segment length, which is why short inputs are unaffected.

### The fix

Wrap `interpolate` so the length is computed with a small epsilon tolerance
before the ceiling, then delegate to the original implementation with an
explicit `size` (the buggy path only runs when called with `scale_factor`):

```python
val = float(input.shape[i + 2]) * float(scale_factor[i])
rounded = round(val, 9)
if abs(val - rounded) < 1e-9:        # 988.0000000000001 -> 988.0
    val = rounded
size.append(max(1, int(math.ceil(val))))
```

`kokoro/istftnet.py` does `from ..interpolate import interpolate`, so it holds
its own reference — the patch rebinds **both** `interpolate_mod.interpolate` and
`istftnet.interpolate`. The patch is idempotent and best-effort (a failure logs
a warning and leaves TTS otherwise running). On success you'll see this line the
first time TTS loads:

```text
[INFO] app.patches.mlx_audio_kokoro — Applied mlx-audio interpolate ceil-drift patch.
```

### Verification

Reproduces the exact numbers from the traceback:

| | down (×1/300) | up (×300) |
|---|---|---|
| Before patch | 296400 → 989 | → 296700 ❌ |
| After patch  | 296400 → 988 | → 296400 ✅ |

```python
import mlx.core as mx
from mlx_audio.tts.models import interpolate as interp
from app.patches import patch_interpolate_ceil_drift

x = mx.zeros((1, 1, 296400))
patch_interpolate_ceil_drift()
down = interp.interpolate(x, scale_factor=1/300, mode="linear")
up   = interp.interpolate(down, scale_factor=300,  mode="linear")
assert down.shape[2] == 988 and up.shape[2] == 296400
```

### Removal

Delete `app/patches/mlx_audio_kokoro.py` (plus its export in
`app/patches/__init__.py` and the `patch_interpolate_ceil_drift()` call in
`_ensure_tts_loaded()`) once `mlx-audio` releases a version that computes the
resample length without the float64 ceil drift (e.g. restores `mx.ceil` or adds
its own epsilon tolerance). Re-run the verification above against the new version
before removing.

- Upstream regression introduced in commit `aaf5ee6`
  ("Fix Kokoro usage from worker threads").
- File: `mlx_audio/tts/models/interpolate.py` (the `math.ceil(...)` in `interpolate`).

---

## 2. mlx-vlm — Gemma 4 unused KV-shared weights on `format: mlx` repos

| | |
|---|---|
| **Dependency** | `mlx-vlm==0.6.3` (latest at time of writing) |
| **Where applied** | `app/patches/mlx_vlm_gemma4.py` → `patch_gemma4_shared_kv_load()`, called before `mlx_vlm.load()` in **both** load paths: `app/services/media_inference_service.py::load()` (API) and `app/services/media_chat_session.py::_load_runtime()` (CLI) |
| **Patched symbol** | `mlx_vlm.models.gemma4.gemma4.Model.load_weights` (overrides the copy inherited from `mlx.nn.Module`) |
| **Status** | Stopgap — remove once the model upload or mlx-vlm is fixed (see *Removal*) |

### Symptom

`POST /api/v1/models/load` with a Gemma 4 E4B repo (e.g.
`mlx-community__gemma-4-e4b-it-bf16`) returns **500**. The server log shows:

```text
ValueError: Received 54 parameters not in model:
language_model.model.layers.24.self_attn.k_norm.weight,
language_model.model.layers.24.self_attn.k_proj.weight,
language_model.model.layers.24.self_attn.v_proj.weight,
… (layers 24–41)
```

### Root cause

Some Gemma 4 checkpoints use **KV-sharing**: a `config.json` with
`num_kv_shared_layers: N` (N > 0) means the last N layers reuse earlier layers'
keys/values, so mlx-vlm builds them with `kv_shared_only=True` and never allocates
their `k_proj`/`v_proj`/`k_norm` modules — the checkpoint's copies are dead weight.
For `gemma-4-e4b-it` that is 42 layers with 18 shared (indices 24–41) → 18 × 3 =
**54** tensors. (`gemma-4-31b-it` and `gemma-4-26b-a4b-it` set
`num_kv_shared_layers: 0`, so they have none and the patch does nothing to them.)

mlx-vlm's own `LanguageModel.sanitize()` strips exactly these, but in
`mlx_vlm/utils.py::load_model` the whole sanitize block is gated on
`if not is_mlx_format:`. This repo's safetensors metadata is `{'format': 'mlx'}`,
so mlx-vlm treats it as already-sanitized, **skips the strip**, and the 54
redundant tensors hit the strict `model.load_weights(...)`, which rejects them.
It is a packaging bug in the upload (flagged "MLX-format / no sanitization
needed" but the redundant weights were never stripped). mlx 0.31.2 / mlx-vlm
0.6.3 / mlx-lm 0.31.3 are all the latest on PyPI — upgrading does not help.

### The fix

Override the gemma4 top-level `Model.load_weights` to drop the unused KV-shared
tensors before delegating to the original loader, reusing mlx-vlm's own predicate
`LanguageModel._is_unused_shared_kv_weight` against the built model:

```python
def load_weights(self, weights, strict=True):
    lm = getattr(self, "language_model", None)
    if isinstance(weights, list) and lm is not None and hasattr(lm, "_is_unused_shared_kv_weight"):
        weights = [(k, v) for (k, v) in weights if not lm._is_unused_shared_kv_weight(k)]
    return original(self, weights, strict=strict)
```

This drops exactly what upstream's sanitize would, so the loaded model is
identical to a correctly-packaged repo — **no accuracy/performance change**. The
shared layers reuse KV from their source layers (whose projections load normally)
and never call `k_proj(x)`/`v_proj(x)`, so the dropped tensors have no module to
load into and never enter the forward pass. Because the predicate reads each
model's own `num_kv_shared_layers`, the patch is **generic across all gemma4
repos** — it strips the right layers for any KV-sharing model (quantized too,
matching `.weight`/`.scales`/`.biases`) and is a **no-op** for non-sharing models
(drops 0). It is idempotent (guarded by a `_pp_gemma4_kv_patch` marker) and
best-effort (a failure logs a warning). On success you'll see this the first time
a media model loads:

```text
[INFO] app.patches.mlx_vlm_gemma4 — Applied mlx-vlm gemma4 KV-shared weight-strip patch.
```

### Verification

```python
from app.patches import patch_gemma4_shared_kv_load
import mlx_vlm

patch_gemma4_shared_kv_load()
model, processor = mlx_vlm.load("models/downloaded/mlx-community__gemma-4-e4b-it-bf16", lazy=True)
# loads without ValueError; logs "dropped 54 unused shared-KV tensors" worth of weights
```

| | before patch | after patch |
|---|---|---|
| `mlx_vlm.load(...)` | `ValueError: Received 54 parameters not in model` | loads OK |

End-to-end: `POST /api/v1/models/load` → 200, then a multimodal chat returns
output. No-regression: non-sharing gemma4 repos (`gemma-4-31b-it`,
`gemma-4-26b-a4b-it`) still load with 0 dropped, and the global
`nn.Module.load_weights` is left untouched so every other model type is unaffected.

### Removal

Delete `app/patches/mlx_vlm_gemma4.py` (plus its export in
`app/patches/__init__.py` and the two `patch_gemma4_shared_kv_load()` call sites —
`MediaInferenceService.load()` and `MediaChatSession._load_runtime()`) once either
the `mlx-community` repo is re-uploaded
with the redundant shared-KV weights stripped, or mlx-vlm runs its shared-KV
sanitize even when `is_mlx_format` is true. Re-run the verification above against
the new version before removing.

- File: `mlx_vlm/utils.py` (the `if not is_mlx_format:` gate around the sanitize
  block in `load_model`).
- Stripping logic that should have run: `mlx_vlm/models/gemma4/language.py`
  (`LanguageModel.sanitize` → `_is_unused_shared_kv_weight`).
