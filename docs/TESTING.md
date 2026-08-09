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

### 4. Verify the audio endpoints

Needs the speech weights on disk (`task audio:setup`) and the server running
(`task run:api`). Use any short clip at `./tmp/clip.wav`.

```bash
# What the frontend gets: models, voices, language codes
curl -s http://127.0.0.1:8000/v1/audio/models | jq '.stt.models, .tts.voices[:3]'

# Each configured STT model, one at a time
for m in $(curl -s http://127.0.0.1:8000/v1/audio/models | jq -r '.stt.models[].id'); do
  echo "── $m"
  curl -s -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
    -F "file=@./tmp/clip.wav;type=audio/wav" -F "model=$m"
  echo
done

# Bad names are 400s, not 503s
curl -s -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@./tmp/clip.wav;type=audio/wav" -F "model=not-a-model"

# TTS with a non-default voice + language code
curl -s -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Testing one two three.","voice":"bf_emma","lang_code":"b"}' \
  --output ./tmp/reply.wav && afplay ./tmp/reply.wav
```

Verify:
- every model in `stt.models` reports `ready: true` after `task audio:setup`
- each transcription returns plausible text, and the server logs one
  `Loading STT model …` line per model switch
- `loaded: true` in `GET /v1/audio/models` tracks the model just used
- after `STT_IDLE_TIMEOUT_SECONDS`, the log shows `Unloading idle STT model …`
  and the next request reloads
- the unknown model returns `400` listing the configured repos, with no mention
  of `audio:setup`

## Notes

- Manual verification is intentionally separate from `task test`.
- If `chat-media` is part of the check, make sure `mlx-vlm` is installed.
- If a model looks suspicious in `model:list`, run `task model:doctor` before
  debugging the chat paths.
