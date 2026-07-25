# Project Flow

Runtime behavior of the AI Service — *what happens when* a request or command
runs. For structure and the reasoning behind it, see
[system-design.md](system-design.md).

## Two entry points, shared core

- **API:** `uvicorn app.main:app` → FastAPI. Started via `task run:api` or
  `python -m app.cli.main serve`.
- **CLI:** `python -m app.cli.main <command>` (Typer). Wrapped by `task ...`.

Both call the same objects in `app/services/`. **They run as separate
processes** and do NOT share loaded models or chat history — only the on-disk
`models/` state (registry + runtime markers) is common.

## Flow 1 — API chat completion (`POST /v1/chat/completions`)

File: [app/api/routes_openai.py](../app/api/routes_openai.py)

1. **Middleware** ([api/middleware.py](../app/api/middleware.py)) assigns a
   `request_id`, reads incoming `X-Correlation-ID`, and logs the request (base64
   media payloads are summarized, not dumped).
2. **Validation** — body parsed into `OpenAIChatCompletionRequest`.
   `_reject_unsupported_chat_features()` returns `400` for `tools`, `logprobs`,
   non-zero penalties, audio *output*, unknown fields, etc. Then
   `_reject_unsupported_media_inputs()` enforces the OpenAI media contract, which
   is **not symmetric**: `image_url.url` may be a `data:image/<subtype>;base64,…`
   URI (payload must decode) **or** an `http(s)://` URL, while `input_audio.data`
   must be decodable base64 — the `input_audio` part has no URL form. Filesystem
   paths are `400` in both cases, as are empty/missing `url` / `data` values (it
   reads the **raw** content parts, not `image_inputs()` / `audio_inputs()`, which
   silently drop empties — otherwise an empty url would answer from text alone).
   Error messages never echo the payload, so a corrupt multi-megabyte body can't
   be dumped into the response or the logs.
   (The CLI bypasses this route and keeps its file-path support.)
3. **Backend selection** — the request goes to the media backend if
   `_request_uses_vlm(messages)` (message carries image/audio parts) OR
   `_model_is_vlm(model_name)` (registry `backend == "mlx-vlm"`); otherwise the
   text backend.
   - Text → `_ensure_model_loaded()` uses `inference_service`.
   - Media → `_ensure_media_model_loaded()` uses `media_inference_service`.
   - Each ensure-loaded call **unloads the other backend first** (one model at a
     time), then loads the requested model if not already resident.
4. **Generation** through the shared `LoadedModelService` interface:
   - non-stream → `_collect_chat_completion()` → one buffered `(text, usage)`,
     returned as `OpenAIChatCompletionResponse`.
   - stream → SSE `StreamingResponse` emitting `data: {chunk}` frames, then
     `data: [DONE]`. If the client disconnects, iteration stops.
5. **Stop sequences** — `_normalize_stop_sequences` + `_split_at_stop_sequence`
   trim output at the first `stop` match (streaming and buffered).
6. **Verbose** — when `verbose: true`, `x_metrics` (timings, load duration) is
   attached to the non-stream body or the final SSE chunk.
7. **Errors** — services raise domain exceptions; handlers in
   [app/main.py](../app/main.py) map them to HTTP + log once by `request_id`.
   These handlers only fire *before* the response starts, so a failure mid-SSE
   is caught and logged inside the stream generator instead (it emits an error
   frame + `[DONE]` to the client).

## Flow 2 — model load/unload (`/api/v1/models`)

File: [app/api/routes_models.py](../app/api/routes_models.py)

- `GET /api/v1/models` → `ModelManager` scans `downloaded/` + `custom/`, merges
  registry timestamps with live runtime markers, and returns state (`ready` /
  `downloading` / `running` / `unsupported` / `incomplete`), `backend`,
  `input_modalities`, and `max_context_tokens`.
- `POST /api/v1/models/load` → loads into the matching backend singleton
  (unloading the other).
- `POST /api/v1/models/unload` → unloads the current model, clears its runtime
  marker.

## Flow 3 — audio (`/v1/audio/*`)

File: [app/api/routes_audio.py](../app/api/routes_audio.py) → `AudioService`.

- `POST /v1/audio/transcriptions` — multipart audio → Whisper → `{"text": ...}`.
  Needs `ffmpeg` for non-WAV uploads.
- `POST /v1/audio/speech` — JSON `{input, voice}` → Kokoro → WAV bytes.
- Both models load lazily on first request alongside the chat model. STT stays
  resident (its handle lives inside `mlx_whisper`); TTS unloads after
  `tts_idle_timeout_seconds` of inactivity and reloads on the next request. No
  chat model emits audio: a spoken reply = model text → TTS endpoint.

## Flow 4 — CLI interactive chat (`chat` / `cli`)

File: [app/cli/main.py](../app/cli/main.py) →
[app/cli/chat_session.py](../app/cli/chat_session.py)

1. CLI verifies the model exists locally (via `ModelManager`).
2. A `ChatSession` is created with its **own in-process `InferenceService`**
   (separate from the API server's singleton).
3. Model loads into memory; the session keeps conversation history in memory.
4. Each user message is appended to history and streamed back token-by-token.
5. `Ctrl+C` during generation stops only the current reply; the session stays
   open.
- `cli` command = `select.py` arrow-key picker → then the same chat loop. Flags
  after `--` are forwarded.

## Flow 5 — CLI media chat (`chat-media`)

File: [app/cli/media_chat_session.py](../app/cli/media_chat_session.py) →
`MediaInferenceService` (`mlx-vlm`). Preload `IMAGE=`/`AUDIO=` or load later with
`/image` and `/audio` inside the session. Preloading checks the model's advertised
`input_modalities` and fails early on a mismatch.

## Flow 6 — model management CLI

`models list | doctor | download | update | delete`, `audio prepare`, all via
`ModelManager`.
- `download` → `snapshot_download` into `models/downloaded/`, registry updated,
  `downloading` marker while writing.
- `doctor` → diagnostics: `model_type`, mapped MLX backend, state, input modes,
  missing files, support verdict, recommendation.
- `update` / `delete` are **blocked** while a model is `running` or
  `downloading` (marker files, cross-process safe).

## Logging & error observability (cross-cutting)

Goal: **every failure is logged exactly once, with a traceback, at the boundary
of its entry point** — so you can always answer *what happened and how*. Setup is
`setup_logging()` (stdout, INFO); get a logger with `get_logger(__name__)`.
Config lives in [app/core/logging.py](../app/core/logging.py).

### Request-scoped ids on every log line

`LoggingMiddleware` sets `correlation_id` and `request_id` **context variables**
([app/core/logging.py](../app/core/logging.py)) at the start of each request and
resets them when it ends. A **log-record factory** stamps those ids onto every
record at creation time, and the formatter prepends
`[correlation_id=… request_id=…]` whenever they're set — so **every** log emitted
while a request is handled is tagged, *including* the framework-free `services/`
logs (e.g. `Model '…' loaded successfully.`), which can't see `request.state`.
Contextvars propagate into the request's task, so this needs no id parameters on
service methods — **don't** add any. Outside a request (CLI, startup) both ids
default to `-` and the segment is omitted, so terminal logs stay clean.

Every boundary relies on the automatic prefix — don't re-embed ids in those
messages. The one that needs a nudge is the **unhandled-500 handler**: it runs in
Starlette's outermost `ServerErrorMiddleware`, *after* `LoggingMiddleware` reset
the context vars, so it re-publishes them from `request.state`
(`correlation_id_var.set(...)` / `request_id_var.set(...)`) before logging to get
the same prefix.

### The invariant: log once, at the boundary

`app/services/` **never logs errors** — it raises domain exceptions from
[core/exceptions.py](../app/core/exceptions.py) (`from exc` preserves the root
cause). The delivery layer logs at its boundary. Each entry point has exactly
one boundary, so a failure is never logged twice and never lost:

| Boundary | Where | Logs |
|---|---|---|
| API, before response starts | exception handlers in [app/main.py](../app/main.py) | the failure with `request_id` + `exc_info` (5xx/unexpected → `error`, 4xx/422 → `warning`), then the outgoing error body as `Response sent` — so the flow stays symmetric with success |
| API, mid-SSE (response already started) | the stream generator in [routes_openai.py](../app/api/routes_openai.py) | `except Exception` → `error` with `request_id` + `exc_info`, then emits an error frame + `[DONE]` |
| CLI command | `_abort()` in [cli/main.py](../app/cli/main.py) | `error`; `exc_info` when handling an exception, message-only for validation aborts |
| CLI chat loop (per-turn) | terminal `except` blocks in the chat sessions | `error` (load/generation) / `warning` (bad media path), with `exc_info` |
| Request in / response out | `LoggingMiddleware` + `response.py` | `info`; base64 media summarized, keyed by `correlation_id` + `request_id` |

### How to add logging when you write or change code

- **New service / business logic (`app/services/`)** — do **not** log the error.
  Raise a domain exception (add one to `core/exceptions.py` if none fits) and
  wrap the cause with `raise MyError(...) from exc`. `info`-level lifecycle logs
  (model loaded/unloaded/downloaded) are fine; error logging is the boundary's job.
- **New API route** — let exceptions propagate to the `main.py` handlers; they
  log. Only `try/except` when you need to translate a domain exception into a
  specific `HTTPException(status_code=...)` — don't log in the route. Return
  success via `send_response(request, ...)`.
- **New streaming (SSE) endpoint** — the `main.py` handlers can't catch a failure
  once the stream has started, so wrap the generator body in
  `except Exception as exc`, log with the request id and traceback
  (`logger.error("... request_id=%s ...", get_request_id(request), exc_info=exc)`),
  then emit a client-visible error frame. This generator is the boundary.
- **New CLI command** — catch the domain exceptions you expect and call
  `_abort(str(exc))` (which logs once). Inside an interactive loop, a terminal
  `except` that prints and continues must first
  `logger.error(..., exc_info=exc)` — otherwise that failure is invisible.
- **Levels:** `error` = 5xx / unexpected / a failed operation; `warning` = 4xx /
  expected-but-notable degradation (e.g. a fallback path taken); `info` =
  lifecycle + request/response; `debug` = best-effort fallbacks you don't want in
  normal logs. Always pass `exc_info` (the exception or `True`) at a boundary so
  the traceback is captured.
- **Don't** add an error log inside `services/` to "make sure it's logged" — the
  boundary already logs it; you'll just double-log the API path.

## Model lifecycle & storage

- Chat models: `models/downloaded/` (registry-tracked) and `models/custom/`
  (scanned, untracked).
- Speech models: `models/hf-cache/` (HF cache, `HF_HUB_CACHE` pinned here).
- Runtime markers: `models/runtime/` — `downloading` / `running`, PID-checked so
  crashes don't leave stale markers.
- Names are sanitized: `org/Repo-Name` on disk becomes `org__Repo-Name`; the
  registry maps it back to the HF repo id.

## Where to make common changes

| Task | Start here |
|------|-----------|
| Change/added an HTTP route | `app/api/routes_*.py` |
| Change OpenAI request acceptance rules | `_reject_unsupported_chat_features` in `routes_openai.py` — then update [openai-compatibility.md](openai-compatibility.md) |
| Change what media input forms the HTTP API accepts (data URI / URL / base64) | `_reject_unsupported_media_inputs` in `routes_openai.py` — then update [openai-compatibility.md](openai-compatibility.md) |
| Change how backend is chosen | `_request_uses_vlm` / `_model_is_vlm` in `routes_openai.py` |
| Change model load/unload behavior | `app/services/base.py` + backend subclass |
| Change text generation / prompt format | `app/services/inference.py` |
| Change multimodal decoding | `app/services/media_inference.py` |
| Change registry / doctor / download | `app/services/model_manager.py` |
| Change a CLI command | `app/cli/main.py` (+ session files for chat loops) |
| Change request/response logging | `app/api/middleware.py` + `app/api/response.py` |
| Add / change error or failure logging | Log at the entry point's boundary (see "Logging & error observability"): `main.py` handlers, the SSE generator in `routes_openai.py`, or `_abort` in `cli/main.py` — never in `services/` |
| Add a config/env value | `app/config.py` (`Settings`) |
| Add a request/response field | `app/schemas/*.py` |
