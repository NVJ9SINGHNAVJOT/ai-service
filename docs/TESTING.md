# Testing Guide

## Automated Tests

Run the automated suite only:

```bash
task test
```

Or directly:

```bash
python3 -m pytest tests/ -v
```

These tests are designed to be deterministic and should not require a downloaded
local model, `./tmp/testing.jpg`, or an interactive terminal session.

## Manual Local Verification

Use this flow when you want to verify real local-model behavior after automated
tests pass.

### 1. Diagnose the model

```bash
task model:doctor MODEL=<local-model-name>
```

Check:
- the model state looks healthy for the scenario you want to test
- `Loadable` is correct
- the `Inputs` hint matches expected media support

### 2. Verify text chat with verbose output

```bash
task model:chat MODEL=<local-model-name> -- --verbose
```

Prompt it with a short message and verify:
- the reply completes cleanly
- control returns to the `You:` prompt
- the verbose block prints after the response
- no leaked turn markers like `<end_of_turn>` appear

### 3. Verify image chat with a real file

Use the local image at `./tmp/testing.jpg`:

```bash
task model:chat-media MODEL=<local-model-name> IMAGE=./tmp/testing.jpg
```

Verify:
- the image loads successfully
- the assistant answers about the image
- the session remains interactive after the first answer

## Notes

- Manual verification is intentionally separate from `task test`.
- If `chat-media` is part of the check, make sure `mlx-vlm` is installed.
- If a model looks suspicious in `model:list`, run `task model:doctor` before
  debugging the chat paths.
