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
- Support streaming in both CLI and API
- Auto-load models on demand when an inference request arrives
- Keep the current API model loaded until it is swapped, explicitly unloaded, or the server stops

## Requirements

- macOS on Apple Silicon
- Python 3.11 or newer
- [Task](https://taskfile.dev) optional but recommended

Why Apple Silicon matters:

The project uses `mlx-lm`, which is built for Apple's MLX stack and is intended for Apple Silicon hardware.

## Project Layout

```text
app/
├── api/                          # FastAPI routers (thin HTTP layer)
│   ├── routes_health.py          #   GET  /health
│   ├── routes_models.py          #   GET/POST /api/v1/models[...]
│   └── routes_openai.py          #   POST /v1/chat/completions
├── cli/
│   └── main.py                   # Typer CLI: models / chat / chat-media / serve
├── core/
│   ├── exceptions.py             # Domain exception hierarchy
│   └── logging.py                # Logging setup
├── schemas/
│   ├── inference.py              # Chat + OpenAI request/response models
│   └── model.py                  # Model-management payloads
├── services/                     # Business logic
│   ├── base_inference_service.py #   LoadedModelService ABC (shared lifecycle)
│   ├── inference_service.py      #   Text inference (mlx-lm)
│   ├── media_inference_service.py#   Multimodal inference (mlx-vlm)
│   ├── chat_session.py           #   Interactive terminal text chat
│   ├── media_chat_session.py     #   Interactive terminal image/audio chat
│   ├── model_manager.py          #   Download / list / update / delete / registry
│   └── model_runtime_state.py    #   Cross-process "downloading"/"running" markers
├── config.py                     # Settings (env / .env) and resolved paths
└── main.py                       # App factory, CORS, routers, shared singletons

models/
├── downloaded/                   # Models fetched from Hugging Face
├── custom/                       # Manually placed model folders
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
- `base_inference_service.py` defines `LoadedModelService`, the abstract base that owns the shared model lifecycle (lock, loaded-model state, load timing, runtime marker, and generation kwargs). Both inference services extend it so the OpenAI route can treat either backend through one interface
- `inference_service.py` loads a text model with `mlx-lm` and performs generate/chat calls
- `media_inference_service.py` loads a multimodal model with `mlx-vlm` for image/audio chat completions
- `chat_session.py` runs the interactive terminal text chat loop
- `media_chat_session.py` runs the interactive terminal image/audio chat loop
- `model_runtime_state.py` persists tiny marker files so `downloading` / `running` states are visible across processes

### `app/api/*`

FastAPI route files. These are intentionally thin and mostly:

- validate request bodies
- call service methods
- convert exceptions into HTTP responses

### `app/cli/main.py`

Typer-based command-line interface for people who want to manage models or chat directly from terminal.

### `app/main.py`

Creates the FastAPI app, configures CORS, registers routes, and creates the shared
`inference_service` (text / mlx-lm) and `media_inference_service` (multimodal / mlx-vlm)
singletons that the routes reuse across requests.

## How Model Storage Works

There are two model locations:

- `models/downloaded/`
- `models/custom/`

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
`["text", "image"]`.

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

## Why Model Names Look Like `mlx-community__Llama-3.2-3B-Instruct-4bit`

Hugging Face repo IDs often contain `/`, for example:

```text
mlx-community/Llama-3.2-3B-Instruct-4bit
```

That is converted into a filesystem-safe local name:

```text
mlx-community__Llama-3.2-3B-Instruct-4bit
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

The project is intended for Python 3.11 or newer.

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

Current example values:

```env
MODELS_BASE_DIR=models
DOWNLOADED_MODELS_DIR=models/downloaded
CUSTOM_MODELS_DIR=models/custom
MODEL_REGISTRY_FILE=models/registry.json

API_HOST=0.0.0.0
API_PORT=8000

DEFAULT_MAX_TOKENS=512
DEFAULT_TEMPERATURE=0.7
DEFAULT_TOP_P=0.9
DEFAULT_REPETITION_PENALTY=1.1
```

Optional values:

- `DEFAULT_MODEL`
- `HF_TOKEN`

Use `HF_TOKEN` only if you need access to a gated Hugging Face model.

## Quick Start

### List models

```bash
task model:list
```

### Download a model

```bash
task model:download MODEL=mlx-community/Llama-3.2-3B-Instruct-4bit
```

### Diagnose models

```bash
task model:doctor
task model:doctor MODEL=mlx-community__Llama-3.2-3B-Instruct-4bit
```

`model:doctor` is useful when a model appears in the list but still fails to
load. It reports the model's `model_type`, mapped MLX backend, current state,
estimated input modes such as `text`, `image`, and `audio`, missing files,
whether the installed `mlx_lm` appears to support it, and a short
recommendation.

### Start CLI chat

```bash
task model:chat MODEL=mlx-community__Llama-3.2-3B-Instruct-4bit
```

During generation, press `Ctrl+C` to stop only the current reply. The session
stays open so you can keep chatting.

### Run media chat

```bash
task model:chat-media MODEL=mlx-community__gemma-4-e4b-bf16 IMAGE=/path/to/image.jpg
```

This starts an interactive multimodal chat session. You can preload an image or
audio clip once, then keep typing follow-up questions in the terminal.
It also supports `/image /path/to/new-image.jpg` and `/audio /path/to/new.wav`
if you want to switch media without leaving the session.

You can also start without initial media and load it later from inside the
session:

```bash
task model:chat-media MODEL=mlx-community__gemma-4-e4b-bf16
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

### Model management

```bash
task model:list
task model:doctor
task model:download MODEL=mlx-community/Llama-3.2-3B-Instruct-4bit
task model:update MODEL=mlx-community__Llama-3.2-3B-Instruct-4bit
task model:delete MODEL=mlx-community__Llama-3.2-3B-Instruct-4bit FORCE=true
task model:chat MODEL=mlx-community__Llama-3.2-3B-Instruct-4bit
task model:chat-media MODEL=mlx-community__gemma-4-e4b-bf16 IMAGE=/path/to/image.jpg
task model:chat-media MODEL=mlx-community__gemma-4-e4b-bf16
```

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
python -m app.cli.main models download --repo mlx-community/Llama-3.2-3B-Instruct-4bit
```

### Diagnose models

```bash
python -m app.cli.main models doctor
python -m app.cli.main models doctor --name mlx-community__Llama-3.2-3B-Instruct-4bit
```

The doctor output also shows a color-coded `Inputs` hint so you can quickly
see whether a model looks text-only or multimodal.

### Update a model

```bash
python -m app.cli.main models update --name mlx-community__Llama-3.2-3B-Instruct-4bit
```

### Delete a model

```bash
python -m app.cli.main models delete --name mlx-community__Llama-3.2-3B-Instruct-4bit --force
```

### Chat with a model

```bash
python -m app.cli.main chat --model mlx-community__Llama-3.2-3B-Instruct-4bit
```

### Chat with image or audio

```bash
python -m app.cli.main chat-media \
  --model mlx-community__gemma-4-e4b-bf16 \
  --image /path/to/image.jpg
```

Or start first and load media later with `/image` or `/audio`:

```bash
python -m app.cli.main chat-media --model mlx-community__gemma-4-e4b-bf16
```

```bash
python -m app.cli.main chat-media \
  --model <local-model-name> \
  --audio /path/to/audio.wav
```

Optional chat settings:

```bash
python -m app.cli.main chat \
  --model mlx-community__Llama-3.2-3B-Instruct-4bit \
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

The API has two main groups of routes:

- health
- model management

The inference surface is OpenAI-compatible chat completions.

Text-only requests run through `mlx-lm`. Requests that include image or audio
content in OpenAI-style multimodal message parts are routed through `mlx-vlm`
automatically.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/api/v1/models` | List all local models |
| POST | `/api/v1/models/load` | Load a model into memory |
| POST | `/api/v1/models/unload` | Unload the currently loaded model |
| POST | `/v1/chat/completions` | OpenAI-compatible chat completions |

## Health Endpoint

```bash
curl http://127.0.0.1:8000/health
```

Example response:

```json
{
  "status": "ok",
  "model_loaded": false,
  "loaded_model": null
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
    "name": "mlx-community__Llama-3.2-3B-Instruct-4bit"
  }'
```

### Unload a model

```bash
curl -X POST http://127.0.0.1:8000/api/v1/models/unload \
  -H "Content-Type: application/json" \
  -d '{}'
```

## Sending Chat History

To preserve conversation context, send the whole relevant `messages` array each time.

Example:

```json
{
  "model": "mlx-community__Llama-3.2-3B-Instruct-4bit",
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

Example image request:

```json
{
  "model": "mlx-community__gemma-4-e4b-bf16",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image in detail."},
        {"type": "image_url", "image_url": {"url": "/absolute/path/to/image.jpg"}}
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
mlx-community__Llama-3.2-3B-Instruct-4bit
```

Current compatibility rules for `/v1/chat/completions`:

- implemented: `model`, `messages`, `developer` / `system` / `user` / `assistant` roles, `max_tokens`, `max_completion_tokens`, `temperature`, `top_p`, `repetition_penalty`, `stream`, `stop`, `n=1`, text/image/audio user inputs
- accepted and ignored: `store`, `metadata`, `service_tier`, `seed`, `safety_identifier`, `stream_options`, `user`
- rejected with `400`: `tools`, `tool_choice`, `parallel_tool_calls`, `function_call`, `prediction`, audio output config via `audio` or non-text `modalities`, `logprobs`, `top_logprobs`, non-zero `frequency_penalty`, non-zero `presence_penalty`, unknown extra fields

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
    model="mlx-community__Llama-3.2-3B-Instruct-4bit",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one short sentence."},
    ],
)

print(response.choices[0].message.content)
```

### OpenAI Python Image Example

```python
from openai import OpenAI

client = OpenAI(
    api_key="test-key",
    base_url="http://127.0.0.1:8000/v1",
)

response = client.chat.completions.create(
    model="mlx-community__gemma-4-e4b-bf16",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "/absolute/path/to/image.jpg"}},
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
  model: "mlx-community__Llama-3.2-3B-Instruct-4bit",
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
    "model": "mlx-community__Llama-3.2-3B-Instruct-4bit",
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

Every endpoint in Swagger UI (`/docs`) ships interactive **Examples** that mirror the
Postman collection in [postman/ai-service.postman_collection.json](/Users/navjot/Desktop/GitRepos/ai-service/postman/ai-service.postman_collection.json) —
including text, image, audio, streaming, verbose, and `stop` scenarios, plus the negative
cases that return `400`. Pick one from the *Examples* dropdown and press **Try it out**.
Replace placeholder model names, image paths, and audio data with values that exist on
your machine before sending.

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
5. `app/services/inference_service.py`
6. `app/services/chat_session.py`
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
