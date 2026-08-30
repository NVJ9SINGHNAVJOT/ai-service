# System Design

Diagrams only. For the written breakdown — directory map, naming conventions, and
what each component does — see [structure.md](structure.md).

## Reading the numbers

Edges that represent a **step in a flow** are numbered so a path can be followed
in order. Edges that represent **structure** (an import, "depends on") are never
numbered — that difference is the point of the notation.

| Situation | Notation | Read it as |
|---|---|---|
| Linear step | `1`, `2`, `3` | happens next |
| **Split — either/or** (one branch is taken) | same number, letter suffix: `2a`, `2b` | *or* |
| **Split — fan-out** (every branch is taken) | dotted decimal: `4.1`, `4.2` | *and* |
| **Converge** (branches rejoin) | the shared number repeats on both incoming edges | both paths arrive at the same step |
| Structural / dependency edge | no label | not part of any flow |

So when an arrow reaches a box and then splits in two, the question is whether the
flow picks one exit or takes both. Picks one → `2a` / `2b` (they are the *same*
step, two outcomes, so the counter does **not** advance twice). Takes both →
`4.1` / `4.2` (one step, two effects). When the branches meet again, the next
number is written once on each incoming arrow, not renumbered per branch:

```mermaid
flowchart LR
    a["step"] -->|1| b["branch point"]
    b -->|2a| c["taken when X"]
    b -->|2b| d["taken when not X"]
    c -->|3| e["rejoin"]
    d -->|3| e
    e -->|"4.1"| f["both happen"]
    e -->|"4.2"| g["both happen"]
```

Two diagram types opt out:

- **Sequence diagrams** use mermaid's `autonumber`, which counts messages
  linearly. Branching is already expressed by the `alt` / `else` blocks, so the
  letter suffixes are redundant there — an `alt` block visually brackets its own
  alternative.
- **State diagrams** and the **module dependency graph** have no step order at
  all: their edges are transitions and imports, not a sequence. They stay
  unnumbered.

## Layers

Two delivery mechanisms over one framework-free service core. Numbers trace one
request; `1a`/`1b` are the two entry points, `3a`–`3d` the backend a request
lands on (exactly one per request).

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
        whisper["mlx-whisper<br/>STT · Whisper"]
        audio["mlx-audio<br/>STT · Parakeet<br/>TTS · Kokoro"]
    end

    disk[("models/<br/>weights · registry.json · runtime markers")]

    client -->|1a| api
    user -->|1b| cli
    api -->|2| services
    cli -->|2| services
    services -->|3a| mlxlm
    services -->|3b| mlxvlm
    services -->|3c| whisper
    services -->|3d| audio
    services -->|4| disk
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
        conc["concurrency.py<br/>chat gate + chat thread"]
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
    appmain --> conc
    appmain --> inf & minf & aud

    ro --> resp
    ro --> conc
    rm --> conc
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
    participant G as chat gate<br/>(api/concurrency.py)
    participant T as InferenceService<br/>(mlx-lm)
    participant V as MediaInferenceService<br/>(mlx-vlm)
    participant H as Exception handlers<br/>(app/main.py)

    C->>M: POST /v1/chat/completions
    M->>M: set correlation_id + request_id<br/>log request (media summarized)
    M->>R: forward

    R->>R: parse OpenAIChatCompletionRequest
    R->>R: _reject_unsupported_chat_features() → 400
    R->>R: _reject_unsupported_media_inputs() → 400

    R->>G: acquire_chat_gate(chat_queue_timeout_seconds)
    alt another chat still generating
        G--)R: TimeoutError
        R-->>C: 503 server busy
    else gate acquired
        G-->>R: held through load + generation
    end

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
        R->>T: chat_stream() — pumped via aiter_chat()
        loop per token
            T-->>R: delta
            R-->>C: data: {chunk}
        end
        R-->>C: data: [DONE]<br/>(x_metrics on final chunk if verbose)
    end

    R->>G: release (the SSE generator's finally when streaming)

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
        backend is resident, in its own two
        slots (one STT, one TTS), each with
        its own idle-unload timer.
    end note
```
