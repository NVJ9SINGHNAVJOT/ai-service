# OpenAI Compatibility

What `POST /v1/chat/completions` supports, ignores, and rejects — the request
contract for anyone pointing an OpenAI SDK at this server.

Source of truth in code:

- [app/schemas/inference.py](../app/schemas/inference.py) —
  `OpenAIChatCompletionRequest` (declared fields + range validation),
  `ChatMessage` (multimodal content parts).
- [app/api/routes_openai.py](../app/api/routes_openai.py) —
  `_reject_unsupported_chat_features` (the 400 rules), `_IGNORED_CHAT_FIELDS`
  (accept-but-drop set), `_reject_unsupported_media_inputs` (media contract).

The request model is `extra="allow"`: unknown fields are captured, then any
unknown field **not** in `_IGNORED_CHAT_FIELDS` is rejected with HTTP 400. So the
default answer for "does it support X?" is **no, loudly** — you get an error
rather than a silently ignored setting.

For the runtime path a request takes after validation (backend selection,
streaming, stop sequences), see
[project-flow.md — Flow 1](project-flow.md#flow-1--api-chat-completion-post-v1chatcompletions).

---

## ✅ Supported & honored

| Field | Notes |
|---|---|
| `model` | Required. Local (sanitized) model name; auto-loaded, swapping out any other loaded model. |
| `messages` | Required, min length 1. Supports multimodal content parts (see below). |
| `max_tokens` | 1–32768. |
| `max_completion_tokens` | 1–32768. OpenAI alias; folded into `max_tokens` when that is unset. |
| `temperature` | 0.0–2.0. |
| `top_p` | 0.0–1.0. |
| `n` | Only the value `1`. Anything else → 400. |
| `stream` | `true` → Server-Sent Events; frames end with `data: [DONE]`. |
| `stop` | String or list (max 4, non-empty). Output trimmed before the stop sequence. |
| `repetition_penalty` | 1.0–2.0. **Non-standard extension** (not an OpenAI field). |
| `verbose` | `true` → server timing metrics in `x_metrics`. **Non-standard extension.** |

### Conditionally supported (only specific values; else 400)

| Field | Allowed | Otherwise |
|---|---|---|
| `response_format` | `null` or `{"type": "text"}` | any other `type` → 400 |
| `modalities` | `null` or `["text"]` | any non-`text` entry → 400 |

### Multimodal message content

A message whose `content` is a list of parts can carry media. Any image/audio
part routes the request to the `mlx-vlm` backend automatically.

| Content part | Accepted |
|---|---|
| `{"type": "text" \| "input_text", ...}` | ✅ |
| `{"type": "image_url", ...}` / `{"type": "input_image", ...}` | ✅ image input |
| `{"type": "input_audio", "input_audio": {"data": <base64>, "format": "wav"}}` | ✅ audio input |

The accepted *forms* are **not symmetric** (enforced by
`_reject_unsupported_media_inputs`):

- `image_url.url` — a `data:image/<subtype>;base64,…` URI (payload must decode)
  **or** an `http(s)://` URL.
- `input_audio.data` — decodable base64 only; there is no URL form.
- Filesystem paths are 400 for both, as are empty/missing `url` / `data` values.
  Error messages never echo the payload, so a corrupt multi-megabyte body is
  never dumped into the response or the logs.

The CLI (`chat-media`) bypasses this route and *does* accept local file paths.

---

## 🟡 Accepted but ignored (no effect)

These pass validation and are silently dropped, so ordinary OpenAI SDK calls
don't break on boilerplate fields.

| Field | Notes |
|---|---|
| `user` | Opaque end-user id; accepted, unused. |
| `metadata` | In `_IGNORED_CHAT_FIELDS`. |
| `store` | Ignored. |
| `service_tier` | Ignored. |
| `seed` | Ignored — generation is **not** seeded or deterministic. |
| `safety_identifier` | Ignored. |
| `stream_options` | Ignored (e.g. `include_usage` has no effect). |
| `frequency_penalty` | Accepted only at `0`/`null` (no-op). Any non-zero value → 400. |
| `presence_penalty` | Accepted only at `0`/`null` (no-op). Any non-zero value → 400. |
| `logprobs` | Accepted only when falsy/`null`. `true` → 400. |

---

## ⛔ Rejected (HTTP 400)

`_reject_unsupported_chat_features` returns `400 — "...is not supported..."` (or a
field-specific message):

| Field / condition | Reason |
|---|---|
| `tools` | Tool calling not supported yet. |
| `tool_choice` | Not supported yet. |
| `parallel_tool_calls` | Not supported yet. |
| `function_call` | Not supported yet. |
| `prediction` | Not supported yet. |
| `audio` | Endpoint does not generate audio output — use `/v1/audio/speech`. |
| `top_logprobs` (not null) | Not supported yet. |
| `logprobs` truthy | Not supported yet. |
| `frequency_penalty` ≠ 0/null | Not supported yet. |
| `presence_penalty` ≠ 0/null | Not supported yet. |
| `n` > 1 | Only a single choice supported. |
| `stop` empty, or > 4 sequences | Invalid stop spec. |
| `response_format.type` ∉ {null, text} | Only text output. |
| `modalities` with a non-text entry | Only `["text"]`. |
| `max_tokens` ≠ `max_completion_tokens` (both set) | Must match. |
| **any unknown field** not in the ignore set | e.g. `thinking` → `400 — Unsupported OpenAI chat completions field(s): thinking` |

### Schema-level validation (HTTP 422, from Pydantic)

Out-of-range values fail *before* the custom checks:
`max_tokens` / `max_completion_tokens` ∉ [1, 32768], `temperature` ∉ [0, 2],
`top_p` ∉ [0, 1], `repetition_penalty` ∉ [1, 2], `n` < 1, empty `messages`.

---

## Response shape

`OpenAIChatCompletionResponse`: `id`, `object`, `created`, `model`, one `choices`
entry (`index`, `message{role, content}`, `finish_reason`), and `usage`
(`prompt_tokens`, `completion_tokens`, `total_tokens`). `x_metrics` is `null`
unless the request sent `verbose: true`.

Streaming emits `data: {chunk}` SSE frames terminated by `data: [DONE]`; with
`verbose: true` the metrics ride on the final chunk.

---

## Changing this contract

Adding or relaxing a field means editing **both** source-of-truth files —
declare it on `OpenAIChatCompletionRequest` (or add it to `_IGNORED_CHAT_FIELDS`
to accept-and-drop), and adjust `_reject_unsupported_chat_features` — then update
the tables above in the same change. Doc drift here is a bug: this file is the
contract clients read.
