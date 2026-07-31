# System Design

Diagrams only. For the written breakdown — directory map, naming conventions, and
what each component does — see [structure.md](structure.md).

## Layers

Two delivery mechanisms over one framework-free service core.

```mermaid
flowchart TD
    client["HTTP client"]
    user["Terminal user"]

    subgraph delivery["Delivery layers"]
        direction LR
        api["app/api/<br/>FastAPI · Starlette<br/><i>HTTP-only</i>"]
        cli["app/cli/<br/>Typer · Rich<br/><i>terminal-only</i>"]
    end

    subgraph core["Shared core"]
        services["app/services/<br/>pure business logic<br/><b>must not import</b><br/>FastAPI · Starlette · Typer · Rich"]
    end

    subgraph shared["Shared primitives"]
        direction LR
        schemas["app/schemas/<br/>pydantic contracts"]
        corepkg["app/core/<br/>exceptions · logging"]
        config["app/config.py<br/>Settings singleton"]
    end

    subgraph backends["MLX backends"]
        direction LR
        mlxlm["mlx-lm<br/>text"]
        mlxvlm["mlx-vlm<br/>multimodal"]
        whisper["mlx-whisper<br/>STT"]
        audio["mlx-audio<br/>TTS"]
    end

    disk[("models/<br/>weights · registry.json · runtime markers")]

    client --> api
    user --> cli
    api --> services
    cli --> services
    services --> mlxlm
    services --> mlxvlm
    services --> whisper
    services --> audio
    services --> disk
    delivery -.-> shared
    core -.-> shared
```

## Module dependency graph

Who imports whom inside `app/`. Every module also imports from
`app/config.py` + `app/core/` (and most from `app/schemas/`); those edges are
aggregated as the dotted lines at the bottom to keep the graph readable.

```mermaid
flowchart TB
    appmain["app/main.py<br/>app factory + service singletons"]

    subgraph api["app/api/ — HTTP delivery"]
        direction TB
        ro["routes_openai.py"]
        rm["routes_models.py"]
        ra["routes_audio.py"]
        rh["routes_health.py"]
        mw["middleware.py"]
        resp["response.py"]
    end

    subgraph cli["app/cli/ — terminal delivery"]
        direction TB
        cmain["main.py<br/>Typer commands"]
        sel["select.py"]
        cs["chat_session.py<br/>ChatSession"]
        mcs["media_chat_session.py<br/>MediaChatSession"]
    end

    subgraph services["app/services/ — shared core"]
        direction TB
        inf["inference.py<br/>InferenceService"]
        minf["media_inference.py<br/>MediaInferenceService"]
        base["base.py<br/>LoadedModelService (ABC)"]
        aud["audio.py<br/>AudioService"]
        mm["model_manager.py<br/>ModelManager"]
        mrs["model_runtime_state.py<br/>ModelRuntimeState"]
    end

    subgraph patches["app/patches/"]
        direction TB
        pk["mlx_audio_kokoro.py"]
        pv["mlx_vlm_gemma4.py"]
    end

    subgraph shared["app/config.py · app/core/ · app/schemas/"]
        direction LR
        cfg["config.py<br/>Settings"]
        exc["core/exceptions.py"]
        log["core/logging.py"]
        sch["schemas/"]
    end

    appmain --> ro & rm & ra & rh
    appmain --> mw
    appmain --> inf & minf & aud

    ro --> resp
    rm --> resp
    ra --> resp
    rh --> resp
    ro --> mm
    rm --> mm
    ro -. "_strip_audio_data_uri" .-> minf

    ro -. "deferred import<br/>of the singletons" .-> appmain
    rm -. " " .-> appmain
    ra -. " " .-> appmain
    rh -. " " .-> appmain

    cmain --> sel & cs & mcs
    cmain --> mm
    cs --> inf
    mcs --> mrs
    mcs --> pv

    inf --> base
    minf --> base
    minf --> pv
    aud --> pk
    base --> mrs
    mm --> mrs

    api -.-> shared
    cli -.-> shared
    services -.-> shared
    patches -.-> log
```

`MediaChatSession` is deliberately **not** wired to `MediaInferenceService`: it
imports `mlx_vlm` directly via `importlib` and manages its own runtime marker.
Only the HTTP path uses `MediaInferenceService`.

## Request flow — `POST /v1/chat/completions`

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant M as LoggingMiddleware
    participant R as routes_openai.py
    participant T as InferenceService<br/>(mlx-lm)
    participant V as MediaInferenceService<br/>(mlx-vlm)
    participant H as Exception handlers<br/>(app/main.py)

    C->>M: POST /v1/chat/completions
    M->>M: set correlation_id + request_id<br/>log request (media summarized)
    M->>R: forward

    R->>R: parse OpenAIChatCompletionRequest
    R->>R: _reject_unsupported_chat_features() → 400
    R->>R: _reject_unsupported_media_inputs() → 400

    alt _request_uses_vlm(messages) or _model_is_vlm(model)
        R->>T: unload (one model at a time)
        R->>V: _ensure_media_model_loaded()
        V-->>R: loaded handle
    else text request
        R->>V: unload (one model at a time)
        R->>T: _ensure_model_loaded()
        T-->>R: loaded handle
    end

    Note over T,V: Generation below goes to whichever backend was selected<br/>above — both expose the same LoadedModelService interface

    alt stream = false
        R->>T: chat()
        T-->>R: (text, usage)
        R->>R: _split_at_stop_sequence()
        R-->>C: OpenAIChatCompletionResponse<br/>(+ x_metrics if verbose)
    else stream = true
        R->>T: chat_stream()
        loop per token
            T-->>R: delta
            R-->>C: data: {chunk}
        end
        R-->>C: data: [DONE]<br/>(x_metrics on final chunk if verbose)
    end

    Note over R,H: Failure BEFORE the response starts →<br/>handlers in app/main.py log once + map to HTTP
    Note over R: Failure MID-SSE → the stream generator is the<br/>boundary: logs with exc_info, emits error frame + [DONE]
```

## Model lifecycle

States are visible across processes via marker files in `models/runtime/`.

```mermaid
stateDiagram-v2
    [*] --> absent

    absent --> downloading: models download
    downloading --> ready: snapshot_download ok<br/>registry updated
    downloading --> absent: download failed

    ready --> running: load into a backend<br/>API load or CLI chat
    running --> ready: unload · process exit ·<br/>PID check clears a stale marker

    ready --> unsupported: doctor verdict —<br/>model_type unsupported by MLX
    ready --> incomplete: doctor verdict —<br/>required files missing

    ready --> absent: models delete
    note right of running
        update and delete are BLOCKED
        while downloading or running
    end note
```

## Backend mutual exclusion

Loading into one backend releases the other — limited unified memory.

```mermaid
stateDiagram-v2
    [*] --> none

    none --> text: load a text model
    none --> media: load a VLM

    text --> text: swap text model<br/>(unloads previous)
    media --> media: swap VLM<br/>(unloads previous)

    text --> media: media request<br/>(text backend released first)
    media --> text: text request<br/>(media backend released first)

    text --> none: unload
    media --> none: unload

    note right of none
        AudioService (STT/TTS) is independent:
        it loads alongside whichever chat
        backend is resident.
    end note
```
