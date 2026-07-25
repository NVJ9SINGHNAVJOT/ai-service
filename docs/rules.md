# Rules — structure & system design

Start here. This is the canonical, portable entry point for how the AI Service is
built and the rules to preserve when changing it. It lives in `docs/` (not the
tool-specific `.claude/` folder) **on purpose**: this repo is consumed as a
submodule, and `docs/` ships with it — so a parent repo, its tooling, or any
contributor can point at `docs/` directly. Our own `.claude/CLAUDE.md` also refers
here rather than keeping a separate copy.

## Read these first

- **Structure & why** — how the code is organized (layers, directory map, key
  classes, naming conventions, extension points): [system-design.md](system-design.md)
- **Runtime behavior** — what happens per request/CLI flow (backend selection,
  model lifecycle, logging, and a "where to make common changes" table):
  [project-flow.md](project-flow.md)
- **Adding your own models** — [custom-models.md](custom-models.md)
- **Manual model verification** — [TESTING.md](TESTING.md)

Read on demand (only when the task touches that area):

- **OpenAI request contract** — which `/v1/chat/completions` fields are honored,
  ignored, or rejected, and the media-input rules:
  [openai-compatibility.md](openai-compatibility.md)

## The core rule

One shared, framework-free service core, two delivery mechanisms:

```
   HTTP client → app/api/ (FastAPI) ┐
                                     ├─→ app/services/  (pure business logic)
 terminal user → app/cli/ (Typer)  ┘        │
                                            ▼
                    mlx-lm · mlx-vlm · mlx-whisper · mlx-audio
```

- `app/services/` is the shared core. It **must not** import FastAPI, Starlette,
  Typer, or Rich. Both delivery layers call the *same* service objects.
- HTTP-only code lives in `app/api/`; terminal-only code lives in `app/cli/`.

## Cross-cutting invariants

- **One model at a time per backend.** Loading a new model unloads the current
  one; the text and media backends are mutually exclusive (limited unified
  memory).
- **Model names are sanitized.** `org/Repo-Name` on disk becomes `org__Repo-Name`;
  the registry maps the sanitized name back to the HF repo id.
- **Services never raise HTTP.** They raise domain exceptions from
  `app/core/exceptions.py`; `app/api/` translates to status codes, `app/cli/`
  prints them.
- **Errors are logged once, at the boundary** of each entry point (never inside
  `app/services/`). See the logging section in [project-flow.md](project-flow.md).

## Keep the docs in sync

Treat doc drift as a bug. When a change moves a file, alters a runtime flow, or
touches an invariant, update the relevant doc in the **same** change — see the
"when to update which file" rules in
[.claude/rules/docs-maintenance.md](../.claude/rules/docs-maintenance.md).
