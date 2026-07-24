# System Design

Architecture reference for the AI Service. Read this to understand *how the code
is organized and why*; read [project-flow.md](project-flow.md) to understand
*what happens at runtime*.

## Core idea: one shared service core, two delivery mechanisms

This is a local MLX inference app for Apple Silicon. The single most important
structural fact:

```
                 ┌─────────────┐     ┌─────────────┐
   HTTP client → │  app/api/   │     │  app/cli/   │ ← terminal user
                 │ (FastAPI)   │     │  (Typer)    │
                 └──────┬──────┘     └──────┬──────┘
                        │                   │
                        └─────────┬─────────┘
                                  ▼
                          ┌───────────────┐
                          │ app/services/ │  pure business logic
                          │  (framework-  │  (no HTTP, no terminal)
                          │   agnostic)   │
                          └───────┬───────┘
                                  ▼
                   mlx-lm · mlx-vlm · mlx-whisper · mlx-audio
```

`app/services/` is the shared core. It must not import FastAPI, Starlette, Typer,
or Rich. Anything HTTP-only lives under `app/api/`; anything terminal-only lives
under `app/cli/`. Both delivery layers call the *same* service objects.

## Directory layout

```
app/
├── config.py            # Settings (env/.env) + resolved paths; module singleton `settings`
├── main.py              # FastAPI app factory; owns the shared service singletons
│
├── core/                # shared primitives (no framework deps)
│   ├── exceptions.py    #   MLXManagerError hierarchy (ModelNotFoundError, ...)
│   └── logging.py       #   logging setup + get_logger
│
├── schemas/             # pydantic request/response contracts (shared)
│   ├── inference.py     #   chat + OpenAI-compatible bodies, Role enum
│   ├── model.py         #   model-management payloads
│   └── audio.py         #   STT/TTS bodies
│
├── services/            # PURE shared business logic — the core
│   ├── base.py              #   LoadedModelService ABC (shared model lifecycle)
│   ├── inference.py         #   InferenceService — text, mlx-lm
│   ├── media_inference.py   #   MediaInferenceService — multimodal, mlx-vlm
│   ├── audio.py             #   AudioService — STT (mlx-whisper) + TTS (mlx-audio)
│   ├── model_manager.py     #   ModelManager — download/list/update/delete/registry/doctor
│   └── model_runtime_state.py #  ModelRuntimeState — cross-process marker files
│
├── api/                 # HTTP delivery layer (FastAPI/Starlette only)
│   ├── routes_health.py     #   GET  /health
│   ├── routes_models.py     #   /api/v1/models[...]
│   ├── routes_openai.py     #   POST /v1/chat/completions (backend selection lives here)
│   ├── routes_audio.py      #   POST /v1/audio/transcriptions, /v1/audio/speech
│   ├── middleware.py        #   LoggingMiddleware (request-id, correlation-id, body sanitizing)
│   └── response.py          #   send_response / log_response / get_request_id helpers
│
├── cli/                 # terminal delivery layer (Typer/Rich only)
│   ├── main.py              #   Typer app: models / audio / chat / chat-media / cli / serve
│   ├── select.py            #   zero-dep arrow-key model picker
│   ├── chat_session.py      #   ChatSession — interactive text loop (uses InferenceService)
│   └── media_chat_session.py #  MediaChatSession — interactive image/audio loop
│
└── patches/             # runtime monkeypatches for upstream MLX bugs (see PATCHES.md)
    ├── mlx_audio_kokoro.py
    └── mlx_vlm_gemma4.py
```

### Naming conventions
- Files inside `services/` are **not** suffixed with `_service` — the folder
  already says "service", so `inference.py` not `inference_service.py`. This
  matches `schemas/` and `core/`, which don't suffix either.
- `routes_*.py` stay flat inside `api/` (no `api/routes/` subpackage).
- Class names still carry role suffixes where it aids readability
  (`InferenceService`, `ModelManager`, `ChatSession`).

### Why middleware and response.py live under `api/`
They are HTTP-only. `LoggingMiddleware` is a Starlette `BaseHTTPMiddleware` and
runs only inside FastAPI; `response.py` builds HTTP response envelopes and reads
request state. Neither has any meaning for the CLI, so they belong to the API
delivery layer — not a top-level `middleware/` or `utils/` package.

## Key components

### `config.py` — `Settings`
Pydantic-settings singleton (`settings`), loaded from env/`.env`. String path
fields map cleanly to env vars; resolved absolute `Path` objects are exposed via
`*_path` properties. `ensure_directories()` creates the models layout. On import
it also sets `HF_HUB_CACHE` to the project-local cache so HF downloads stay in
`models/hf-cache/` instead of `~/.cache/huggingface`. `example_text_model` /
`example_media_model` supply the model names pre-filled into the Swagger
"Try it out" example bodies (read at import by `routes_openai.py` /
`routes_models.py`), so they change everywhere from one env var.

### `services/base.py` — `LoadedModelService` (ABC)
The heart of the inference design. Owns the **backend-agnostic** lifecycle shared
by both inference backends:
- a reentrant lock guarding load/unload,
- the loaded-model name, the in-memory model handle, last load duration,
- the cross-process "running" marker (via `ModelRuntimeState`),
- `_generation_kwargs(...)` building MLX sampler/logits args from request +
  config defaults.

Subclasses implement the abstract hooks `load`, `chat`, `chat_stream`, and
`_release_backend` (Template Method). Because both subclasses share this
interface, `routes_openai.py` can treat either backend through one substitutable
handle (Liskov) — it just picks the right service and calls the same methods.

**Concurrency invariant:** the lock protects load/unload only. Concurrent
`generate` calls on one loaded model are *not* safe (MLX isn't thread-safe at the
C level). A multi-user server would need a request queue — see Extension points.

### `services/inference.py` — `InferenceService`
Text backend (`mlx-lm`). Loads a model, formats messages via the tokenizer chat
template (falls back to a plain-text format), and runs `generate` /
`stream_generate`. Also handles chat-template stop markers and stop-sequence
splitting.

### `services/media_inference.py` — `MediaInferenceService`
Multimodal backend (`mlx-vlm`). Same interface as `InferenceService`, but decodes
image/audio content parts (including base64 `input_audio` → temp file) before
generation.

### `services/audio.py` — `AudioService`
Local STT (Whisper via `mlx-whisper`) and TTS (Kokoro via `mlx-audio`). Both load
lazily on first use alongside the chat model. STT's handle lives inside
`mlx_whisper` (we don't own it) so it stays resident for the process lifetime; the
Kokoro TTS handle *is* ours, so a re-arming idle timer unloads it after
`tts_idle_timeout_seconds` of inactivity (0 = keep resident) and it reloads on the
next request. Independent of the chat backends.

### `services/model_manager.py` — `ModelManager`
Everything about models *on disk*: download (`snapshot_download`), list, update,
delete, registry (`models/registry.json`) read/write, and the `doctor`
diagnostics (backend detection, `model_type` → MLX support, input-modality
inference, missing-file checks). Stateless w.r.t. loaded models — it reasons about
files, not memory.

### `services/model_runtime_state.py` — `ModelRuntimeState`
Tiny marker files under `models/runtime/` that make `downloading` / `running`
states visible **across processes** (e.g. a CLI chat and the API server are
separate processes and don't share memory). Also does PID-liveness checks so a
crashed process doesn't leave a stale `running` marker.

### `main.py` — app factory + singletons
Creates the three shared service singletons at module load:
`inference_service`, `media_inference_service`, `audio_service`. Wires CORS →
`LoggingMiddleware` → exception handlers (which log once at the boundary,
correlated by `request_id`) → routers. Also patches `app.openapi()` to inject the
chat-completion 200 examples that FastAPI's `exclude_none` would otherwise strip.

## Cross-cutting invariants
- **One model at a time per backend.** Loading a new model unloads the current
  one. The text and media backends are also mutually exclusive: entering one path
  unloads the other (limited unified memory).
- **Model names are sanitized.** HF repo ids like `mlx-community/Llama-...` become
  filesystem-safe `mlx-community__Llama-...`; the registry maps the sanitized name
  back to the repo id. The API uses the sanitized name as the identifier.
- **Services never raise HTTP.** They raise domain exceptions from
  `core/exceptions.py`; `api/` translates those to status codes, `cli/` prints
  them.
- **Errors are logged once, at the boundary.** Each entry point has one logging
  boundary: for the API it's the exception handlers in `main.py`, keyed by
  `request_id`; for the CLI it's `_abort()` in `cli/main.py` (plus the terminal
  `except` blocks in the chat sessions). The SSE streaming path is its own
  boundary — once the response has started the `main.py` handlers can no longer
  catch it, so its generator logs failures in-route (see `routes_openai.py`).
  Every boundary logs with a traceback (`exc_info`). Keep this the *only* place
  each failure is logged — don't add error logs inside `services/` or you'll
  double-log the API path.

## Extension points (from README "Recommended Next Improvements")
- API-key auth → add as FastAPI dependency/middleware in `api/`.
- Request queue for safe concurrent inference → wrap `LoadedModelService` in
  `services/`.
- OpenAI `/v1/models`, embeddings → new `routes_*.py` + service method.
- Persistent conversation storage → new service + schema.

## Testing
`tests/` uses `pytest` with fakes — no real model downloads. Tests import services
directly (e.g. `from app.services.inference import InferenceService`) and drive
routes via FastAPI `TestClient`. Manual real-model steps live in
[TESTING.md](TESTING.md).
