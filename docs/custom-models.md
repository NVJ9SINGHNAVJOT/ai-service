# Custom Models

Place each model in its own subfolder here. The folder name becomes the model's
local name in `task model:list` and in API requests.

```
models/custom/
└── my-finetuned-llama/       ← folder name = model name in the service
    ├── config.json            ← required
    ├── tokenizer_config.json  ← required
    ├── tokenizer.json         ← required (or tokenizer.model for sentencepiece)
    ├── special_tokens_map.json
    └── model.safetensors      ← MLX-converted weights (one file or sharded)
```

The service considers a folder valid when it contains both `config.json` and
`tokenizer_config.json`. Missing either will mark the model as `incomplete`.

---

## Minimum required files

| File | Why it is needed |
|------|-----------------|
| `config.json` | Architecture config. Must contain `"model_type"` so the service can detect the backend and check MLX support. |
| `tokenizer_config.json` | Tokenizer metadata. Required for the model to be considered loadable. |
| `tokenizer.json` | Fast tokenizer vocab/merges (used by most modern models). |
| `*.safetensors` | The actual weights. Can be a single file or sharded (`model-00001-of-00002.safetensors` + `model.safetensors.index.json`). |

---

## Example config.json (text-only Llama-style model)

```json
{
  "model_type": "llama",
  "hidden_size": 2048,
  "intermediate_size": 8192,
  "num_hidden_layers": 16,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "vocab_size": 32000,
  "max_position_embeddings": 4096,
  "rms_norm_eps": 1e-5,
  "rope_theta": 10000.0,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.40.0"
}
```

The `model_type` value is the key field. It must match an architecture that the
installed `mlx_lm` runtime supports. Common supported values: `llama`, `mistral`,
`qwen2`, `gemma`, `phi`, `starcoder2`, `falcon`.

Run `task model:doctor MODEL=<your-folder-name>` after placing files to see
whether the service considers the model ready.

---

## How to convert a fine-tuned HuggingFace model to MLX

If you fine-tuned with HuggingFace Transformers and have a checkpoint in
`/path/to/finetuned-model`, convert it with:

```bash
python -m mlx_lm.convert \
  --hf-path /path/to/finetuned-model \
  --mlx-path models/custom/my-finetuned-model
```

This writes the weights as `.safetensors` and copies all tokenizer and config
files into the output folder. No further steps needed — the folder is ready to
use.

Optional: quantize to 4-bit during conversion to reduce memory usage:

```bash
python -m mlx_lm.convert \
  --hf-path /path/to/finetuned-model \
  --mlx-path models/custom/my-finetuned-model-4bit \
  --quantize \
  --q-bits 4
```

---

## How to fuse LoRA adapters into the base model

If you fine-tuned with `mlx_lm.lora` and have adapter weights in `adapters/`:

```bash
python -m mlx_lm.fuse \
  --model /path/to/base-model \
  --adapter-path adapters/ \
  --save-path models/custom/my-fused-model
```

This merges the adapter weights into the base model and saves the result as a
standalone MLX model folder, ready to drop into `models/custom/`.

---

## Verify the model works

```bash
# Check that the service sees it and considers it ready
task model:doctor MODEL=my-finetuned-model

# Quick raw inference test (bypasses the service, uses mlx_lm directly)
.venv/bin/python -m mlx_lm.generate \
  --model models/custom/my-finetuned-model \
  --prompt "Hello, who are you?" \
  --max-tokens 64

# Start a full chat session through the service
task model:chat MODEL=my-finetuned-model
```

---

## Notes

- Custom models are never tracked in `models/registry.json`. They are
  discovered by scanning this folder on every `model:list` call.
- `task model:update` does not apply to custom models. Replace the files
  manually if you want to update weights.
- Deletion via `task model:delete` is blocked for custom models by default.
  Pass `FORCE=true` and `ALLOW_CUSTOM=true` if you want the CLI to remove
  the folder, or just delete it manually from here.
