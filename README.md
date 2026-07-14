# AI Service

A local AI service for Apple Silicon Macs, built on top of MLX-LM.

This project gives you three things in one codebase:

- a CLI to download, list, update, delete, and chat with local MLX models
- a FastAPI server for model management and OpenAI-compatible inference
- an OpenAI-compatible chat endpoint so other apps can call your local models using common SDKs

It is designed for local development and learning, while still keeping the code structured enough to grow into a more production-like service.

## What This Project Does

This app helps you run MLX-compatible LLMs on a Mac with Apple Silicon.

At a high level:

1. You download or place a model on disk
2. The app keeps track of available models
3. The CLI or API loads one model into memory
4. You send prompts or chat messages
5. The model responds either as a full response or as a stream

This is useful if you want:

- a local chat backend for your app
- an OpenAI-style API in front of your MLX models
- a simple learning project to understand how model serving works
- a single-machine inference server for personal tools or internal testing

## Who This Is For

This project is a good fit if you are:

- learning Python and want a readable backend project
- building a Node.js, Python, or Go app that should talk to local models
- using Apple Silicon and want MLX-native inference
- experimenting with chat APIs and streaming responses

This project is not trying to be a full distributed inference platform. It is intentionally simpler:

- one machine
- one loaded model at a time
- local filesystem storage
- simple JSON registry

## Main Features

- Download MLX-compatible Hugging Face models
- Keep a registry of downloaded models
- Support custom local model folders
- Start interactive terminal chat
- Run a FastAPI-based OpenAI-compatible inference API
- Expose an OpenAI-compatible `/v1/chat/completions` endpoint
- Run fully local speech-to-text and text-to-speech through OpenAI-compatible `/v1/audio/*` endpoints (Whisper + Kokoro on MLX)
- Support streaming in both CLI and API
- Auto-load models on demand when an inference request arrives
- Keep the current API model loaded until it is swapped, explicitly unloaded, or the server stops

## Requirements

- macOS on Apple Silicon
- Python 3.13 or newer
- [Task](https://taskfile.dev) optional but recommended

Why Apple Silicon matters:

The project uses `mlx-lm`, which is built for Apple's MLX stack and is intended for Apple Silicon hardware.

## Project Layout

```text
app/
├── api/                          # HTTP delivery layer (FastAPI/Starlette only)
│   ├── routes_health.py          #   GET  /health
│   ├── routes_models.py          #   GET/POST /api/v1/models[...]
│   ├── routes_openai.py          #   POST /v1/chat/completions
│   ├── routes_audio.py           #   POST /v1/audio/transcriptions, /v1/audio/speech
│   ├── middleware.py             #   LoggingMiddleware (request-id / correlation-id / body summary)
│   └── response.py               #   send_response / log_response / get_request_id helpers
├── cli/                          # Terminal delivery layer (Typer/Rich only)
│   ├── main.py                   #   Typer CLI: models / audio / chat / chat-media / cli / serve
│   ├── select.py                 #   Zero-dep arrow-key picker used by the interactive `cli` command
│   ├── chat_session.py           #   Interactive terminal text chat
│   └── media_chat_session.py     #   Interactive terminal image/audio chat
├── core/
│   ├── exceptions.py             # Domain exception hierarchy
│   └── logging.py                # Logging setup
├── schemas/
│   ├── inference.py              # Chat + OpenAI request/response models
│   ├── model.py                  # Model-management payloads
│   └── audio.py                  # STT / TTS request/response models
├── services/                     # Pure shared business logic (no HTTP, no terminal)
│   ├── base.py                   #   LoadedModelService ABC (shared lifecycle)
│   ├── inference.py              #   Text inference (mlx-lm)
│   ├── media_inference.py        #   Multimodal inference (mlx-vlm)
│   ├── audio.py                  #   Speech-to-text (mlx-whisper) + text-to-speech (mlx-audio)
│   ├── model_manager.py          #   Download / list / update / delete / registry
│   └── model_runtime_state.py    #   Cross-process "downloading"/"running" markers
├── config.py                     # Settings (env / .env) and resolved paths
└── main.py                       # App factory, CORS, routers, shared singletons

models/
├── downloaded/                   # Chat models fetched from Hugging Face
├── custom/                       # Manually placed model folders
├── hf-cache/                     # HuggingFace cache for speech models (Whisper/Kokoro)
├── runtime/                      # Transient activity marker files
└── registry.json                # Metadata for downloaded models

tests/
├── conftest.py
├── test_cli_chat.py
├── test_media_chat.py
├── test_logging.py
├── test_models.py
└── test_openai_api.py
```

## Architecture Overview

The codebase is split into a few clear layers.

### `app/config.py`

Loads settings from environment variables or `.env`, and resolves paths.

### `app/core/*`

Shared project utilities:

- `exceptions.py` defines project-specific exceptions
- `logging.py` sets up application logging

### `app/schemas/*`

Pydantic models for request and response validation.

These define the shapes of:

- model-management payloads
- inference requests
- OpenAI-compatible request and response bodies

### `app/services/*`

This is where the main business logic lives.

- `model_manager.py` handles downloading, registry updates, listing, updating, and deleting models
- `base.py` defines `LoadedModelService`, the abstract base that owns the shared model lifecycle (lock, loaded-model state, load timing, runtime marker, and generation kwargs). Both inference services extend it so the OpenAI route can treat either backend through one interface
- `inference.py` loads a text model with `mlx-lm` and performs generate/chat calls
- `media_inference.py` loads a multimodal model with `mlx-vlm` for image/audio chat completions
- `audio.py` runs local speech-to-text (Whisper via `mlx-whisper`) and text-to-speech (Kokoro via `mlx-audio`); both load lazily alongside the chat model — STT stays resident, while TTS unloads after `TTS_IDLE_TIMEOUT_SECONDS` of inactivity and reloads on the next request
- `model_runtime_state.py` persists tiny marker files so `downloading` / `running` states are visible across processes

The interactive terminal chat loops (`chat_session.py`, `media_chat_session.py`)
live under `app/cli/`, not here — they are CLI-only and are covered below.

### `app/api/*`

The HTTP delivery layer. Route files (`routes_*.py`) are intentionally thin and mostly:

- validate request bodies
- call service methods
- convert exceptions into HTTP responses

This layer also holds the HTTP-only cross-cutting helpers: `middleware.py`
(request logging + request/correlation ids) and `response.py` (response
envelope + logging helpers). They live here rather than in a top-level package
because they only ever run inside the FastAPI server.

### `app/cli/*`

Typer-based command-line interface for people who want to manage models or chat
directly from terminal. Alongside `main.py` and the `select.py` picker, this is
where the interactive chat loops live (`chat_session.py`, `media_chat_session.py`)
since they are terminal-only.

### `app/main.py`

Creates the FastAPI app, configures CORS, registers routes, and creates the shared
`inference_service` (text / mlx-lm), `media_inference_service` (multimodal / mlx-vlm),
and `audio_service` (STT/TTS) singletons that the routes reuse across requests.

## How Model Storage Works

There are two chat-model locations:

- `models/downloaded/`
- `models/custom/`

Speech models (Whisper for STT, Kokoro for TTS) are cached separately in the
project-local HuggingFace cache:

- `models/hf-cache/`

This keeps every download inside the project instead of the global
`~/.cache/huggingface`. Override it with `HF_CACHE_DIR` (or an explicit
`HF_HUB_CACHE`).

There is also a runtime activity directory:

- `models/runtime/`

This stores tiny marker files used to expose live states in `task model:list`,
such as `downloading` and `running`.

### Downloaded models

These are models fetched through the app from Hugging Face using `huggingface_hub.snapshot_download()`.

They are tracked in:

- `models/registry.json`

The registry stores metadata such as:

- local name
- original Hugging Face repo id
- absolute path
- timestamps

When you run `task model:list`, the CLI combines registry timestamps with live
runtime state and shows:

- `State`: `ready`, `downloading`, `running`, `unsupported`, or `incomplete`
- `Loadable`: whether the model should be considered safe to load right now
- `Created`: when the model was first registered
- `Updated`: when the model was last registered or refreshed

The `/api/v1/models` endpoint returns the same core metadata, including a
best-effort `input_modalities` field such as `["text"]` or
`["text", "image"]`, and a best-effort `max_context_tokens` field (the model's
context window, read from `config.json`; `null` when it can't be determined).

The states mean:

- `ready`: the model directory looks complete and the installed `mlx_lm` runtime appears to support its `model_type`
- `downloading`: files are still being written, so the model is not treated as loadable yet
- `running`: another process currently has the model loaded
- `unsupported`: the model files exist, but the installed `mlx_lm` runtime does not support that architecture
- `incomplete`: the directory is missing expected files such as `config.json` or tokenizer metadata

To help prevent data loss or confusing runtime behavior:

- `update` and `delete` are blocked while a model is `running` or `downloading`
- chat/API load paths fail early for `unsupported` or incomplete models with a clearer error message

### Custom models

These are models you place manually into:

- `models/custom/`

Custom models are listed by scanning that folder. They are not tied to a registry entry in the same way downloaded models are.

## Why Model Names Look Like `mlx-community__Meta-Llama-3.1-8B-Instruct-8bit`

Hugging Face repo IDs often contain `/`, for example:

```text
mlx-community/Meta-Llama-3.1-8B-Instruct-8bit
```

That is converted into a filesystem-safe local name:

```text
mlx-community__Meta-Llama-3.1-8B-Instruct-8bit
```

This matters because:

- folders cannot safely use arbitrary remote IDs as-is
- the API uses local model names as identifiers
- the registry maps the sanitized name back to the original repo id

## Setup

Fastest option:

```bash
task setup
source .venv/bin/activate
```

This will:

1. clean common Python cache files
2. create a local `.venv`
3. install dependencies into that local environment

### 1. Create a virtual environment

```bash
task venv
source .venv/bin/activate
```

This creates a project-local `.venv` so this repo stays isolated from your Mac's global Python packages and from other projects on your machine.

If you want to run it manually instead of Task:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

The project is intended for Python 3.13 or newer.

### 2. Install dependencies

Using Task:

```bash
task install
```

Or directly with pip:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### 3. Create your environment file

```bash
cp .env.example .env
```

### 4. Review `.env`

The `.env` file allows you to override the default configuration. All values have sensible defaults built into the code, so you only need to uncomment and set the ones you want to change.

```env
# ── API server (defaults: 0.0.0.0:8000) ─────────────────────────
API_PORT=8000

# ── Inference defaults ───────────────────────────────────────────
# These act as fallbacks if the API caller doesn't specify them.
DEFAULT_MAX_TOKENS=512
DEFAULT_TEMPERATURE=0.7
DEFAULT_TOP_P=0.9
DEFAULT_REPETITION_PENALTY=1.1

# ── Swagger example models ───────────────────────────────────────
# Model names pre-filled into the Swagger "Try it out" example bodies
# (chat completions + model load/unload). Set these to models you have
# locally so the examples are ready to send. Takes effect on restart.
EXAMPLE_TEXT_MODEL=org__your-text-model
EXAMPLE_MEDIA_MODEL=org__your-media-model

# ── HuggingFace (required for gated/private models) ─────────────
HF_TOKEN=hf_...

# ── Speech models (STT / TTS) ───────────────────────────────────
STT_MODEL=mlx-community/whisper-large-v3-turbo
TTS_MODEL=prince-canuma/Kokoro-82M
TTS_VOICE=af_heart
TTS_LANG_CODE=a
TTS_IDLE_TIMEOUT_SECONDS=60   # unload the TTS model after N idle seconds (0 = keep resident)

# ── Model cache (where HF weights download; default: project-local) ──
HF_CACHE_DIR=models/hf-cache
```

## Quick Start

### List models

```bash
task model:list
```

### Download a model

```bash
task model:download MODEL=org/your-text-model
```

### Diagnose models

```bash
task model:doctor
task model:doctor MODEL=org__your-text-model
```

`model:doctor` is useful when a model appears in the list but still fails to
load. It reports the model's `model_type`, mapped MLX backend, current state,
estimated input modes such as `text`, `image`, and `audio`, missing files,
whether the installed `mlx_lm` appears to support it, and a short
recommendation.

### Interactive CLI (pick a model, then chat)

```bash
task cli
```

This lists your loadable models and lets you move the highlight with the
↑/↓ arrow keys (or `j`/`k`), press `Enter` to select, and drop straight into a
chat session — no need to remember the sanitised model name. Press `q` or
`Ctrl+C` to cancel the picker.

Chat flags are passed through after `--`:

```bash
task cli -- --verbose --temperature 0.5
```

### Start CLI chat

```bash
task model:chat MODEL=org__your-text-model
```

During generation, press `Ctrl+C` to stop only the current reply. The session
stays open so you can keep chatting.

### Run media chat

```bash
task model:chat-media MODEL=org__your-media-model IMAGE=/path/to/image.jpg
```

This starts an interactive multimodal chat session. You can preload an image or
audio clip once, then keep typing follow-up questions in the terminal.
It also supports `/image /path/to/new-image.jpg` and `/audio /path/to/new.wav`
if you want to switch media without leaving the session.

You can also start without initial media and load it later from inside the
session:

```bash
task model:chat-media MODEL=org__your-media-model
```

You can preload audio too:

```bash
task model:chat-media MODEL=<local-model-name> AUDIO=/path/to/audio.wav
```

### Start the API server

```bash
task run:api
```

### Run tests

```bash
task test
```

## Taskfile Commands

The project includes a `Taskfile.yaml` with convenience commands.

### Install

```bash
task setup
```

Or step by step:

```bash
task venv
source .venv/bin/activate
task install
task install:dev
```

### Run the server

```bash
task run:api
```

You can override host and port:

```bash
task run:api API_HOST=127.0.0.1 API_PORT=9000
```

### Run the CLI help

```bash
task run:cli
```

### Interactive CLI

```bash
task cli
task cli -- --verbose --temperature 0.5
```

Launches the arrow-key model picker, then starts a chat with the selected
model. Anything after `--` is forwarded to the chat session as flags.

### Model management

```bash
task model:list
task model:doctor
task model:download MODEL=org/your-text-model
task model:update MODEL=org__your-text-model
task model:delete MODEL=org__your-text-model FORCE=true
task model:chat MODEL=org__your-text-model
task model:chat-media MODEL=org__your-media-model IMAGE=/path/to/image.jpg
task model:chat-media MODEL=org__your-media-model
```

### Speech models (STT / TTS)

```bash
task audio:setup
```

Pre-downloads the Whisper (STT) and Kokoro (TTS) weights (~1.8 GB) into
`models/hf-cache/` so the first voice request is instant. `task setup` runs this
automatically as its final step.

### Testing

```bash
task test
```

Manual local-model verification steps live in [TESTING.md](/Users/navjot/Desktop/GitRepos/ai-service/TESTING.md).

### Cleanup

```bash
task clean
```

## CLI Usage

If you prefer not to use Task:

```bash
python -m app.cli.main --help
```

### List models

```bash
python -m app.cli.main models list
```

### Download a model

```bash
python -m app.cli.main models download --repo org/your-text-model
```

### Diagnose models

```bash
python -m app.cli.main models doctor
python -m app.cli.main models doctor --name org__your-text-model
```

The doctor output also shows a color-coded `Inputs` hint so you can quickly
see whether a model looks text-only or multimodal.

### Update a model

```bash
python -m app.cli.main models update --name org__your-text-model
```

### Delete a model

```bash
python -m app.cli.main models delete --name org__your-text-model --force
```

### Prepare speech models

```bash
python -m app.cli.main audio prepare
```

Pre-downloads the STT (Whisper) and TTS (Kokoro) weights into `models/hf-cache/`.

### Pick a model interactively, then chat

```bash
python -m app.cli.main cli
```

Lists loadable models, lets you select one with the ↑/↓ arrow keys and `Enter`,
then starts a chat. It accepts the same optional chat flags as `chat`
(`--system`, `--max-tokens`, `--temperature`, `--top-p`, `--repetition-penalty`,
`--verbose`).

### Chat with a model

```bash
python -m app.cli.main chat --model org__your-text-model
```

### Chat with image or audio

```bash
python -m app.cli.main chat-media \
  --model org__your-media-model \
  --image /path/to/image.jpg
```

Or start first and load media later with `/image` or `/audio`:

```bash
python -m app.cli.main chat-media --model org__your-media-model
```

```bash
python -m app.cli.main chat-media \
  --model <local-model-name> \
  --audio /path/to/audio.wav
```

Optional chat settings:

```bash
python -m app.cli.main chat \
  --model org__your-text-model \
  --system "You are a helpful coding assistant." \
  --max-tokens 512 \
  --temperature 0.7 \
  --top-p 0.9 \
  --repetition-penalty 1.1
```

### Start the API server

```bash
python -m app.cli.main serve --host 0.0.0.0 --port 8000
```

## How CLI Chat Works

When you run `chat`:

1. The CLI verifies that the model exists locally
2. A `ChatSession` object is created
3. The model is loaded into memory
4. The session keeps a conversation history in memory
5. Each new user message is appended to that history
6. The model response is streamed back to the terminal token by token

This gives you a local interactive chat experience without needing the HTTP API.

Important:

- CLI chat uses its own in-process `InferenceService`
- the API server uses a separate shared `InferenceService`
- they do not share the same loaded model instance or chat history across processes

For image or audio prompts, use `chat-media` instead. It is an interactive
mlx-vlm-powered wrapper intended for multimodal models.

Use:

- `task model:chat` for normal text-only terminal chat
- `task model:chat-media` for image/audio + text prompts, with or without preloaded media

## API Overview

The API has these groups of routes:

- health
- model management
- OpenAI-compatible chat completions
- OpenAI-compatible audio (speech-to-text and text-to-speech)

Text-only requests run through `mlx-lm`. Requests that include image or audio
content in OpenAI-style multimodal message parts are routed through `mlx-vlm`
automatically. Speech transcription and synthesis run through `mlx-whisper` and
`mlx-audio`.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/api/v1/models` | List all local models with state, backend, and input modalities |
| POST | `/api/v1/models/load` | Load a model into memory |
| POST | `/api/v1/models/unload` | Unload the currently loaded model |
| POST | `/v1/chat/completions` | OpenAI-compatible chat completions |
| POST | `/v1/audio/transcriptions` | Speech-to-text (Whisper) — multipart audio → text |
| POST | `/v1/audio/speech` | Text-to-speech (Kokoro) — text → WAV audio |

## Health Endpoint

```bash
curl http://127.0.0.1:8000/health
```

Example response:

```json
{
  "status": "ok",
  "model_loaded": false,
  "loaded_model": null,
  "loaded_model_backend": null
}
```

If a media model is currently loaded for image or audio requests, `loaded_model` will
show that model too.

## Model Management API

### List models

```bash
curl http://127.0.0.1:8000/api/v1/models
```

### Load a model

```bash
curl -X POST http://127.0.0.1:8000/api/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "name": "org__your-text-model"
  }'
```

### Unload a model

```bash
curl -X POST http://127.0.0.1:8000/api/v1/models/unload \
  -H "Content-Type: application/json" \
  -d '{}'
```

## Voice: Speech-to-Text & Text-to-Speech

Two local, OpenAI-compatible audio endpoints run fully on-device — no audio or
text ever leaves the machine.

- **STT** — `POST /v1/audio/transcriptions`: a multipart audio upload returns
  `{ "text": ... }`, transcribed by Whisper on MLX (`mlx-whisper`).
- **TTS** — `POST /v1/audio/speech`: a JSON body `{ "input": "...", "voice": "af_heart" }`
  returns WAV audio bytes, synthesized by Kokoro on MLX (`mlx-audio`).

These are independent of chat: no model in this stack emits audio, so a spoken
reply is produced by sending the model's text reply to the TTS endpoint. (Audio
*input* to a multimodal `mlx-vlm` model is a separate feature of the chat
endpoint.)

### Models

The repos are configurable via `.env`; defaults:

- STT: `mlx-community/whisper-large-v3-turbo`
- TTS: `prince-canuma/Kokoro-82M` (voice `af_heart`, American English)

They download on first use into `models/hf-cache/`. Pre-fetch them so the first
request doesn't block on a multi-GB download:

```bash
task audio:setup          # or: python -m app.cli.main audio prepare
```

### Examples

```bash
# Transcribe a clip (multipart upload)
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@clip.wav;type=audio/wav"
# → {"text": "..."}

# Synthesize speech (returns WAV bytes)
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello from Kokoro.","voice":"af_heart"}' \
  --output reply.wav
```

### Dependencies & notes

- **`ffmpeg`** is required to decode non-WAV uploads (e.g. a browser's webm/opus).
  Install with `brew install ffmpeg`.
- Kokoro's text front-end (`misaki`) can't install its `[en]` extra on Python
  3.13 — its `spacy-curated-transformers` pin has no 3.13 wheels. `requirements.txt`
  instead pins the working English G2P stack explicitly (`misaki` + `spacy` +
  `en_core_web_sm` + `num2words` + `phonemizer-fork` + `espeakng-loader`, which
  bundles espeak-ng so no `brew install espeak-ng` is needed). This matches
  Kokoro's default behavior — no quality difference.

## Sending Chat History

To preserve conversation context, send the whole relevant `messages` array each time.

Example:

```json
{
  "model": "org__your-text-model",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Navjot."},
    {"role": "assistant", "content": "Nice to meet you, Navjot."},
    {"role": "user", "content": "Remind me what my name is."}
  ]
}
```

General rule:

- keep messages in order
- oldest first, newest last
- include prior assistant replies if you want the model to remember them

For text-only requests, each `content` value can stay a plain string.

For image or audio requests, OpenAI-style multimodal `content` arrays are also accepted.
The API currently supports:

- `{"type": "text", "text": "..."}`
- `{"type": "input_text", "text": "..."}`
- `{"type": "image_url", "image_url": {"url": "..."}}`
- `{"type": "input_image", "image_url": "..."}`
- `{"type": "input_audio", "input_audio": {"data": "...", "format": "wav"}}`

Media follows the OpenAI contract, and the two parts are **not** symmetric. A
**filesystem path is never valid** over HTTP and returns a `400`. (The `chat-media` CLI
is separate and still takes file paths.)

For `image_url`, `url` is **either** a **base64 data URI** of the form
`data:<mime>;base64,<bytes>` — e.g.
`"data:image/jpeg;base64," + base64.b64encode(open("photo.jpg", "rb").read()).decode()` —
**or** an `http(s)://` URL, which the server fetches. A `data:` URI whose MIME type is not
`image/*`, or whose base64 does not decode, returns a `400`.

For `input_audio`, `data` must be **base64-encoded audio bytes** (exactly what the OpenAI
SDK sends — the `input_audio` part has no URL form; a `data:audio/...;base64,` URI is also
accepted). `format` is the source type such as `wav` or `mp3`. The server decodes the
base64 to a temporary file before handing it to `mlx-vlm`.

Example image request:

```json
{
  "model": "org__your-media-model",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image in detail."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<base64-encoded-image-bytes>"}}
      ]
    }
  ]
}
```

When an image is present, the server uses `mlx-vlm` instead of `mlx-lm`, so
make sure `mlx-vlm` is installed:

```bash
python -m pip install -U mlx-vlm
```

The same applies to audio-bearing chat completion requests.

## OpenAI Compatibility

The server supports:

- `POST /v1/chat/completions`

This means you can point OpenAI-compatible SDKs at your local server by changing only the `base_url`.

Use your local model name as the `model` value:

```text
org__your-text-model
```

Current compatibility rules for `/v1/chat/completions`:

- implemented: `model`, `messages`, `developer` / `system` / `user` / `assistant` roles, `max_tokens`, `max_completion_tokens`, `temperature`, `top_p`, `repetition_penalty`, `stream`, `stop`, `n=1`, text/image/audio user inputs
- accepted and ignored: `store`, `metadata`, `service_tier`, `seed`, `safety_identifier`, `stream_options`, `user`
- rejected with `400`: `tools`, `tool_choice`, `parallel_tool_calls`, `function_call`, `prediction`, audio output config via `audio` or non-text `modalities`, `logprobs`, `top_logprobs`, non-zero `frequency_penalty`, non-zero `presence_penalty`, unknown extra fields, media inputs that are not a valid OpenAI form (filesystem paths, non-`image/*` data URIs, undecodable base64, empty or missing `url` / `data` values)

This keeps the endpoint friendly to common OpenAI SDK request shapes while
still failing clearly for advanced features that are not implemented here yet.

### Important note about API keys

The local server does not currently enforce API-key authentication, but many SDKs require an API key field anyway.

So you can pass any placeholder value such as:

```text
test-key
```

## OpenAI Python Example

```python
from openai import OpenAI

client = OpenAI(
    api_key="test-key",
    base_url="http://127.0.0.1:8000/v1",
)

response = client.chat.completions.create(
    model="org__your-text-model",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one short sentence."},
    ],
)

print(response.choices[0].message.content)
```

### OpenAI Python Image Example

```python
import base64

from openai import OpenAI

client = OpenAI(
    api_key="test-key",
    base_url="http://127.0.0.1:8000/v1",
)

with open("photo.jpg", "rb") as handle:
    image_data_uri = "data:image/jpeg;base64," + base64.b64encode(handle.read()).decode()

response = client.chat.completions.create(
    model="org__your-media-model",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": image_data_uri}},
            ],
        }
    ],
)

print(response.choices[0].message.content)
```

## OpenAI Node.js Example

```js
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: "test-key",
  baseURL: "http://127.0.0.1:8000/v1",
});

const response = await client.chat.completions.create({
  model: "org__your-text-model",
  messages: [
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: "Say hello in one short sentence." }
  ]
});

console.log(response.choices[0].message.content);
```

## OpenAI-Compatible Streaming

The OpenAI-compatible endpoint uses Server-Sent Events.

If the client disconnects during a streamed response, the server stops
iterating the stream instead of continuing to send chunks to a dead
connection.

### curl example

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-key" \
  -d '{
    "model": "org__your-text-model",
    "messages": [
      {"role": "user", "content": "Write a short haiku about coding."}
    ],
    "stream": true
  }'
```

The response is sent as SSE frames like:

```text
data: {"id":"chatcmpl-...","object":"chat.completion.chunk", ...}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk", ...}

data: [DONE]
```

## Verbose Metrics

If you want server timing details similar to the CLI `--verbose` output, send:

```json
{
  "verbose": true
}
```

On non-streaming requests, the response includes an `x_metrics` object alongside `usage`.

On streaming requests, the final SSE chunk includes `x_metrics`.

## API Docs

When the server is running, FastAPI also exposes built-in docs:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/redoc`
- `http://127.0.0.1:8000/openapi.json`

Swagger UI (`/docs`) is the single source for trying the API. Every endpoint ships
interactive **Examples** — including text, image, audio, streaming, verbose, and
`stop` scenarios, plus the negative cases that return `400`. Pick one from the
*Examples* dropdown and press **Try it out**. For the speech-to-text endpoint, use
the file picker to upload an audio clip directly. Replace placeholder model names,
image paths, and audio data with values that exist on your machine before sending.

## How Inference Works Internally

The main inference flow looks like this:

1. A route receives a request
2. The route ensures the requested model is loaded
3. `InferenceService` formats the prompt
4. `mlx_lm.generate()` or `mlx_lm.stream_generate()` is called
5. The response is returned as either:
   - one full text result
   - streamed chunks

If the same model is requested again, the API reuses the already loaded model.
It does not unload the model automatically after every response.

For chat requests, the message list is converted into a prompt using:

- the tokenizer's chat template if available
- a fallback plain-text format otherwise

## One Model at a Time

This project keeps one model loaded in memory at a time.

Why:

- local Macs have limited unified memory
- model switching is simpler than hosting many models at once
- the implementation stays easier to reason about

If a different model is requested, the current one is unloaded before the next one is loaded.

The currently loaded API model stays in memory until one of these happens:

- you call `POST /api/v1/models/unload`
- another API request needs a different model
- the server shuts down

## Error Handling

The code uses project-specific exceptions such as:

- `ModelNotFoundError`
- `ModelLoadError`
- `InferenceError`
- `DownloadError`
- `RegistryError`
- `InvalidModelPathError`

Routes catch these and convert them into HTTP responses with appropriate status codes.

## Security Notes

This project includes a few basic safety checks:

- model paths are validated to prevent directory traversal
- deletions are restricted to allowed model directories
- custom-model deletion is blocked by default unless explicitly allowed

This is good baseline protection for a local tool, but you should still treat this as a local/dev-oriented service unless you add stronger authentication, authorization, and deployment hardening.

## Current Limitations

- only chat completions are exposed in the OpenAI-compatible layer
- token usage may be estimated when MLX-LM does not provide exact values
- one model is loaded at a time
- no built-in auth enforcement yet
- optimized for local usage, not multi-node deployment

## Testing

Run all tests:

```bash
task test
```

Or:

```bash
python3 -m pytest tests/ -v
```

For real local-model verification with `model:doctor`, verbose text chat, and
image chat using `./tmp/testing.jpg`, see [TESTING.md](/Users/navjot/Desktop/GitRepos/ai-service/TESTING.md).

## Learning Guide for This Codebase

If you are still learning Python, this is a good order to read files:

1. `app/config.py`
2. `app/schemas/model.py`
3. `app/schemas/inference.py`
4. `app/services/model_manager.py`
5. `app/services/inference.py`
6. `app/cli/chat_session.py`
7. `app/api/routes_models.py`
8. `app/api/routes_openai.py`
9. `app/main.py`

Why this order helps:

- config explains app settings
- schemas explain data shapes
- services explain the real logic
- routes show how HTTP is wired on top of that logic
- main shows how the app starts

## Troubleshooting

### The model does not appear in `model:list`

Check:

- the model folder exists under `models/downloaded/` or `models/custom/`
- the folder contains expected files like `config.json` or tokenizer config

If the model still looks wrong, run:

```bash
task model:doctor
```

### A model downloads but will not load

Possible reasons:

- it is not actually an MLX-compatible model
- the download is incomplete
- the model files are not in the expected layout
- the installed `mlx_lm` runtime does not support the model's `model_type`

For example, a model can come from `mlx-community` on Hugging Face and still be
`unsupported` locally if your installed `mlx_lm` version does not ship a loader
for that architecture yet.

To inspect one model directly, run:

```bash
task model:doctor MODEL=<local-model-name>
```

To compare your app against raw MLX behavior, run:

```bash
./.venv/bin/mlx_lm.generate --model /absolute/path/to/model --prompt "Hello" --max-tokens 32
```

If raw `mlx_lm` fails with the same error, the issue is in the installed MLX
runtime rather than this repo's CLI wrapper.

### I want to prompt a model with image or audio

Use the media path rather than the normal text chat path:

```bash
task model:chat-media MODEL=<local-model-name> IMAGE=/path/to/image.jpg
```

If you prefer, you can also omit `IMAGE` and `AUDIO` and load them later with
`/image` or `/audio` from
inside the session:

```bash
task model:chat-media MODEL=<local-model-name>
```

This feature uses `mlx-vlm`, so make sure it is installed:

```bash
python -m pip install -U mlx-vlm
```

You can also provide an optional system prompt:

```bash
task model:chat-media MODEL=<local-model-name> IMAGE=/path/to/image.jpg SYSTEM="You are a helpful media assistant."
```

When you preload `IMAGE` or `AUDIO`, the CLI checks the model's advertised
`input_modalities` first and fails early if the model does not appear to
support that media type.

### Update or delete says the model is busy

This is expected when the model is:

- `running` in another chat or server process
- `downloading` and still being written to disk

Wait for the download to finish, or stop the process currently using the model,
then retry the command.

### The OpenAI SDK connects but responses fail

Check:

- your `base_url` is `http://127.0.0.1:8000/v1`
- the model name is the local sanitized name, not the original HF repo id
- the server is running

### Streaming does not look right in curl

Use:

```bash
curl -N ...
```

The `-N` option disables buffering in curl so streamed output appears as it arrives.

## Development Notes

This project prefers:

- clear services over route-heavy logic
- explicit schemas for request/response bodies
- safe filesystem operations
- simple tests that do not require real model downloads

## Recommended Next Improvements

If you want to grow this project further, the next useful additions would be:

- API-key authentication
- OpenAI-compatible `/v1/models`
- OpenAI-compatible embeddings endpoint if needed
- richer structured logging
- request queueing for safer concurrent inference
- persistent conversation/session storage

## Summary

This project is a clean local serving layer for MLX models on Apple Silicon.

You can:

- manage models from CLI
- chat locally in terminal
- call your models through an OpenAI-compatible API
- use OpenAI-compatible SDKs against your local server
- stream responses in both CLI and API

If you are learning, the codebase is intentionally organized so you can understand it layer by layer without needing to know every advanced Python pattern first.
