# AI Service

A local AI service for Apple Silicon Macs, built on top of MLX-LM.

This project gives you three things in one codebase:

- a CLI to download, list, update, delete, and chat with local MLX models
- a FastAPI server for model management and inference
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
- Run FastAPI-based inference endpoints
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
├── api/
│   ├── routes_health.py
│   ├── routes_models.py
│   ├── routes_inference.py
│   └── routes_openai.py
├── cli/
│   └── main.py
├── core/
│   ├── exceptions.py
│   └── logging.py
├── schemas/
│   ├── inference.py
│   └── model.py
├── services/
│   ├── chat_session.py
│   ├── inference_service.py
│   └── model_manager.py
├── config.py
└── main.py

models/
├── downloaded/
├── custom/
└── registry.json

tests/
├── test_smoke.py
└── test_model_lifecycle.py
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
- `inference_service.py` loads a model and performs generate/chat calls
- `chat_session.py` runs the interactive terminal chat loop

### `app/api/*`

FastAPI route files. These are intentionally thin and mostly:

- validate request bodies
- call service methods
- convert exceptions into HTTP responses

### `app/cli/main.py`

Typer-based command-line interface for people who want to manage models or chat directly from terminal.

### `app/main.py`

Creates the FastAPI app, configures CORS, registers routes, and creates the shared `inference_service` singleton.

## How Model Storage Works

There are two model locations:

- `models/downloaded/`
- `models/custom/`

### Downloaded models

These are models fetched through the app from Hugging Face using `huggingface_hub.snapshot_download()`.

They are tracked in:

- `models/registry.json`

The registry stores metadata such as:

- local name
- original Hugging Face repo id
- absolute path
- timestamps

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

### Start CLI chat

```bash
task model:chat MODEL=mlx-community__Llama-3.2-3B-Instruct-4bit
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
task model:download MODEL=mlx-community/Llama-3.2-3B-Instruct-4bit
task model:update MODEL=mlx-community__Llama-3.2-3B-Instruct-4bit
task model:delete MODEL=mlx-community__Llama-3.2-3B-Instruct-4bit FORCE=true
task model:chat MODEL=mlx-community__Llama-3.2-3B-Instruct-4bit
```

### Testing

```bash
task test
task test:smoke
```

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

## API Overview

The API has three main groups of routes:

- health
- model management
- inference

There is also an OpenAI-compatible layer for chat completions.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/api/v1/models` | List all local models |
| POST | `/api/v1/models/download` | Download a model |
| POST | `/api/v1/models/update` | Update a downloaded model |
| DELETE | `/api/v1/models/{model_name}` | Delete a model |
| POST | `/api/v1/models/load` | Load a model into memory |
| POST | `/api/v1/models/unload` | Unload the currently loaded model |
| POST | `/api/v1/inference/generate` | Prompt-based text generation |
| POST | `/api/v1/inference/chat` | Chat-style inference |
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

## Model Management API

### List models

```bash
curl http://127.0.0.1:8000/api/v1/models
```

### Download a model

```bash
curl -X POST http://127.0.0.1:8000/api/v1/models/download \
  -H "Content-Type: application/json" \
  -d '{
    "repo_id": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "force": false
  }'
```

### Update a model

```bash
curl -X POST http://127.0.0.1:8000/api/v1/models/update \
  -H "Content-Type: application/json" \
  -d '{
    "name": "mlx-community__Llama-3.2-3B-Instruct-4bit"
  }'
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

### Delete a model

```bash
curl -X DELETE "http://127.0.0.1:8000/api/v1/models/mlx-community__Llama-3.2-3B-Instruct-4bit"
```

To allow deleting a custom model:

```bash
curl -X DELETE "http://127.0.0.1:8000/api/v1/models/my-custom-model?allow_custom=true"
```

## Inference API

There are two styles of inference:

- `generate`
- `chat`

### `generate` vs `chat`

Use `generate` when you already have a plain prompt string and want a completion.

Use `chat` when you want role-based messages such as:

- `system`
- `user`
- `assistant`

For most app development, `chat` is the better choice.

## Prompt Generation Endpoint

### Non-streaming request

```bash
curl -X POST http://127.0.0.1:8000/api/v1/inference/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community__Llama-3.2-3B-Instruct-4bit",
    "prompt": "Write one sentence about local AI.",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Streaming request

This endpoint streams newline-delimited JSON.

```bash
curl -N http://127.0.0.1:8000/api/v1/inference/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community__Llama-3.2-3B-Instruct-4bit",
    "prompt": "Write one sentence about local AI.",
    "stream": true
  }'
```

## Chat Endpoint

### Non-streaming request

```bash
curl -X POST http://127.0.0.1:8000/api/v1/inference/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community__Llama-3.2-3B-Instruct-4bit",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Say hello in one sentence."}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Streaming request

```bash
curl -N http://127.0.0.1:8000/api/v1/inference/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community__Llama-3.2-3B-Instruct-4bit",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Write a short haiku about coding."}
    ],
    "stream": true
  }'
```

Example streamed line shape:

```json
{"model":"mlx-community__Llama-3.2-3B-Instruct-4bit","text":"Hello","done":false,"usage":null}
```

Final streamed line includes usage:

```json
{"model":"mlx-community__Llama-3.2-3B-Instruct-4bit","text":" world","done":true,"usage":{"prompt_tokens":12,"completion_tokens":2,"total_tokens":14,"finish_reason":"stop"}}
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

## OpenAI Compatibility

The server supports:

- `POST /v1/chat/completions`

This means you can point OpenAI-compatible SDKs at your local server by changing only the `base_url`.

Use your local model name as the `model` value:

```text
mlx-community__Llama-3.2-3B-Instruct-4bit
```

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

## Streaming Formats Summary

There are two streaming formats in this project.

### Custom API streaming

- endpoint: `/api/v1/inference/chat`
- format: NDJSON
- good for: simple backend consumption and `curl -N`

### OpenAI-compatible streaming

- endpoint: `/v1/chat/completions`
- format: SSE
- good for: OpenAI SDK compatibility

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

Run the fast smoke and lifecycle tests:

```bash
task test:smoke
```

## Learning Guide for This Codebase

If you are still learning Python, this is a good order to read files:

1. `app/config.py`
2. `app/schemas/model.py`
3. `app/schemas/inference.py`
4. `app/services/model_manager.py`
5. `app/services/inference_service.py`
6. `app/services/chat_session.py`
7. `app/api/routes_models.py`
8. `app/api/routes_inference.py`
9. `app/api/routes_openai.py`
10. `app/main.py`

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

### A model downloads but will not load

Possible reasons:

- it is not actually an MLX-compatible model
- the download is incomplete
- the model files are not in the expected layout

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
- call your models through a custom REST API
- use OpenAI-compatible SDKs against your local server
- stream responses in both CLI and API

If you are learning, the codebase is intentionally organized so you can understand it layer by layer without needing to know every advanced Python pattern first.
