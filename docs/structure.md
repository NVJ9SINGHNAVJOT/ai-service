# Structure & Nomenclature

Where everything lives in AI Core, what each piece is responsible for, and
the naming conventions the codebase follows. For the same picture as diagrams,
see [system-design.md](system-design.md).

This is a local MLX inference app for Apple Silicon: one shared, framework-free
service core (`app/services/`) behind two delivery layers — a FastAPI server
(`app/api/`) and a Typer CLI (`app/cli/`).

## Repository layout

Everything that is version-controlled:

```
.
├── app/                     # the application package (mapped below)
├── docs/                    # reference docs — ship with the repo as a submodule
│   ├── structure.md             #   this file: layout, naming, components
│   ├── system-design.md         #   mermaid diagrams
│   ├── openai-compatibility.md  #   /v1/chat/completions request contract
│   ├── custom-models.md         #   dropping your own model into models/custom/
│   └── TESTING.md               #   manual real-model verification
├── tests/                   # pytest suite, fakes only — no real model downloads
│   ├── conftest.py
│   ├── test_audio_api.py
│   ├── test_audio_service.py
│   ├── test_cli_chat.py
│   ├── test_cli_select.py
│   ├── test_concurrency.py
│   ├── test_logging.py
│   ├── test_media_chat.py
│   ├── test_models.py
│   └── test_openai_api.py
├── PATCHES.md               # runtime monkeypatches for upstream MLX bugs
├── README.md                # user-facing guide
├── Taskfile.yaml            # `task ...` command wrappers
├── requirements.txt
├── .env.example             # template for the local .env
├── .gitignore
└── LICENSE
```

### `models/` — created at runtime, never committed

`models/` is **not** in git: every subdirectory is listed in `.gitignore`, and
`Settings.ensure_directories()` ([app/config.py](../app/config.py)) creates the
layout on first use. Expect this on a working checkout:

```
models/
├── downloaded/          # chat models fetched from Hugging Face
├── custom/              # manually placed model folders
├── hf-cache/            # HF cache for speech models (HF_HUB_CACHE points here)
├── runtime/             # transient `downloading` / `running` marker files
└── registry.json        # metadata for downloaded models
```

Other git-ignored directories you may see locally but that are not part of the
repo: `.venv/`, `.claude/` (agent instructions), `plans/`, and `tmp/` (scratch
files — `docs/TESTING.md` uses `tmp/testing.jpg`).

## `app/` package map

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
│   ├── concurrency.py       #   chat gate (asyncio.Lock) + the single chat worker thread
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
    ├── __init__.py          #   re-exports the patch functions — import from here
    ├── mlx_audio_kokoro.py
    └── mlx_vlm_gemma4.py
```

Every package also has an `__init__.py`. All of them are docstring-only except
`patches/__init__.py`, which re-exports `patch_interpolate_ceil_drift` and
`patch_gemma4_shared_kv_load` — callers do `from app.patches import ...`, not
`from app.patches.mlx_vlm_gemma4 import ...`.

## Nomenclature

| Convention | Rule | Example |
|---|---|---|
| Service modules | No `_service` suffix — the folder already says "service". Same for `schemas/` and `core/`. | `services/inference.py`, **not** `inference_service.py` |
| Route modules | `routes_<area>.py`, flat inside `api/` — no `api/routes/` subpackage. | `routes_openai.py`, `routes_audio.py` |
| Classes | Keep role suffixes where it aids readability. | `InferenceService`, `ModelManager`, `ChatSession`, `MediaChatSession`, `ModelRuntimeState` |
| Module-private helpers | `_`-prefixed; they are not part of any module's surface. | `_ensure_model_loaded`, `_reject_unsupported_chat_features`, `_abort` |
| Exceptions | End in `Error`, derive from `MLXManagerError`, live only in `core/exceptions.py`. | `ModelNotFoundError`, `ModelLoadError`, `ModelBusyError` |
| Tests | `tests/test_<area>.py` — named for the area under test, not one file per module. | `test_openai_api.py`, `test_cli_chat.py`, `test_audio_service.py` |
| Model names on disk | HF repo ids are sanitized: `/` becomes `__`. The registry maps back to the repo id. | `mlx-community/Llama-3.2-3B` → `mlx-community__Llama-3.2-3B` |

### Why `middleware.py`, `response.py` and `concurrency.py` live under `api/`

They are HTTP-only. `LoggingMiddleware` is a Starlette `BaseHTTPMiddleware` and
runs only inside FastAPI; `response.py` builds HTTP response envelopes and reads
request state; `concurrency.py` orders *concurrent HTTP requests*, which the CLI
never has — it runs one chat loop per process. None has meaning for the CLI, so
they belong to the API delivery layer — not a top-level `middleware/` or
`utils/` package.

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

**Concurrency invariant:** the lock protects load/unload only, and it does *not*
order requests — it is reentrant, so two callers on the same thread both pass.
Concurrent `generate` calls on one loaded model are not safe (MLX isn't
thread-safe at the C level). Serialization is enforced one layer up, by the chat
gate in [`api/concurrency.py`](../app/api/concurrency.py); the CLI needs none of
it, running a single chat loop per process.

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
Local STT (Whisper via `mlx-whisper`, Parakeet via `mlx-audio`) and TTS (Kokoro
via `mlx-audio`). Models load lazily on first use alongside the chat model, but
never *download*: each load is gated on `_ensure_speech_model_available()`, an
offline HF-cache probe that raises `SpeechModelNotPreparedError` (→ HTTP 503)
when the weights — or every Kokoro voice pack — are absent; a name missing from
an otherwise-populated cache is a client error instead (`InvalidVoiceError`,
`InvalidSTTModelError`, `InvalidLangCodeError` → HTTP 400). `audio prepare` is
the only download path.

Two `_ResidentModel` slots — one STT, one TTS — each hold **one** model behind a
re-arming idle timer that unloads it after `stt_idle_timeout_seconds` /
`tts_idle_timeout_seconds` (0 = keep resident); an in-flight counter keeps the
timer from dropping a model mid-request. Audio requests run concurrently (the
routes use the shared threadpool), so the same counter also gates swaps: callers
wanting the resident model pass straight through, while one wanting a different
model waits on a `Condition` until the in-flight requests drain. A transcription
request picks its model from `settings.available_stt_models` (HF repo ids — *not*
the `org__name` form used for `models/downloaded`), and asking for a different
one unloads the current one once it is safe to. `mlx_whisper` caches its handle
on its own module rather than handing
us one, so the slot primes and clears `mlx_whisper.transcribe.ModelHolder`.
`describe_stt()` / `describe_tts()` back `GET /v1/audio/models` and load nothing.
Independent of the chat backends.

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

## Extension points

- API-key auth → add as FastAPI dependency/middleware in `api/`.
- Multi-request inference (batching, or more than one resident chat model) →
  replace the single gate + single worker thread in `api/concurrency.py`; the
  serialization is deliberate today, not incidental.
- OpenAI `/v1/models`, embeddings → new `routes_*.py` + service method.
- Persistent conversation storage → new service + schema.

## Testing

`tests/` uses `pytest` with fakes — no real model downloads. Tests import services
directly (e.g. `from app.services.inference import InferenceService`) and drive
routes via FastAPI `TestClient`. Manual real-model steps live in
[TESTING.md](TESTING.md).
