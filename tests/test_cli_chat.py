"""
Automated tests for text CLI chat behavior and text inference helpers.
"""

from __future__ import annotations

from pathlib import Path


def test_chat_session_uses_streaming(monkeypatch):
    """ChatSession should build the assistant response from streamed chunks."""
    from app.schemas.inference import ChatMessage, Role
    from app.cli.chat_session import ChatSession

    prompts = iter(["Hello", "quit"])
    printed: list[str] = []

    monkeypatch.setattr("app.cli.chat_session.Prompt.ask", lambda _: next(prompts))
    monkeypatch.setattr("app.cli.chat_session.console.print", lambda *args, **kwargs: printed.append("" if not args else str(args[0])))

    session = ChatSession(model_path=Path("/tmp/fake-model"), model_name="my-model")
    session._history = [ChatMessage(role=Role.system, content="You are helpful.")]
    monkeypatch.setattr(
        session._svc,
        "chat_stream",
        lambda messages, max_tokens=None, temperature=None, top_p=None, repetition_penalty=None: iter(
            [("Hello", None), (" world", {"finish_reason": "stop"})]
        ),
    )

    session._loop()

    assert session._history[-1].role == Role.assistant


def test_chat_session_ctrl_c_stops_current_reply_but_keeps_session(monkeypatch):
    """Ctrl+C during generation should keep the chat open and preserve partial output."""
    from app.schemas.inference import ChatMessage, Role
    from app.cli.chat_session import ChatSession

    prompts = iter(["Hello", "Next", "quit"])
    printed: list[str] = []

    monkeypatch.setattr("app.cli.chat_session.Prompt.ask", lambda _: next(prompts))
    monkeypatch.setattr("app.cli.chat_session.console.print", lambda *args, **kwargs: printed.append("" if not args else str(args[0])))

    session = ChatSession(model_path=Path("/tmp/fake-model"), model_name="my-model")
    session._history = [ChatMessage(role=Role.system, content="You are helpful.")]

    def fake_stream(messages, max_tokens=None, temperature=None, top_p=None, repetition_penalty=None):
        last_user = messages[-1].content
        if last_user == "Hello":
            yield ("Part", None)
            raise KeyboardInterrupt
        yield (" second", {"finish_reason": "stop"})

    monkeypatch.setattr(session._svc, "chat_stream", fake_stream)

    session._loop()

    assistant_messages = [msg.content for msg in session._history if msg.role == Role.assistant]
    assert assistant_messages == ["Part", " second"]
    assert any("Generation stopped" in line for line in printed)


def test_chat_session_verbose_stats(monkeypatch):
    """ChatSession should print verbose timing/token stats when enabled."""
    from app.schemas.inference import ChatMessage, Role
    from app.cli.chat_session import ChatSession

    prompts = iter(["Hello", "quit"])
    printed: list[str] = []

    monkeypatch.setattr("app.cli.chat_session.Prompt.ask", lambda _: next(prompts))
    monkeypatch.setattr("app.cli.chat_session.console.print", lambda *args, **kwargs: printed.append("" if not args else str(args[0])))

    session = ChatSession(model_path=Path("/tmp/fake-model"), model_name="my-model", verbose=True)
    session._history = [ChatMessage(role=Role.system, content="You are helpful.")]
    session._svc._last_load_duration_s = 0.094960042
    monkeypatch.setattr(
        session._svc,
        "chat_stream",
        lambda messages, max_tokens=None, temperature=None, top_p=None, repetition_penalty=None: iter(
            [
                ("Hello", None),
                (
                    " world",
                    {
                        "prompt_tokens": 13,
                        "completion_tokens": 43,
                        "finish_reason": "stop",
                        "metrics": {
                            "total_duration_s": 1.256830584,
                            "prompt_eval_duration_s": 0.267349292,
                            "prompt_eval_rate": 48.63,
                            "eval_duration_s": 0.890170291,
                            "eval_rate": 48.31,
                        },
                    },
                ),
            ]
        ),
    )

    session._loop()

    stats_blocks = [entry for entry in printed if "total duration:" in entry]
    assert len(stats_blocks) == 1
    assert "total duration:       1.256830584s" in stats_blocks[0]
    assert "load duration:        94.960042ms" in stats_blocks[0]
    assert "prompt eval count:    13 token(s)" in stats_blocks[0]
    assert "eval rate:            48.31 tokens/s" in stats_blocks[0]


def test_inference_service_retries_chat_template_with_user_first_messages():
    """User-first tokenizers should keep their native template by folding system text into the first user turn."""
    from app.schemas.inference import ChatMessage, Role
    from app.services.inference import InferenceService

    calls: list[list[dict[str, str]]] = []

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            calls.append(messages)
            if messages[0]["role"] != "user":
                raise ValueError("Conversations must start with a user prompt.")
            return "native prompt"

    svc = InferenceService()
    svc._tokenizer = FakeTokenizer()

    prompt = svc._apply_chat_template(
        [
            ChatMessage(role=Role.system, content="You are helpful."),
            ChatMessage(role=Role.user, content="How are you"),
        ]
    )

    assert prompt == "native prompt"
    assert len(calls) == 2
    assert calls[0][0]["role"] == "system"
    assert calls[1][0]["role"] == "user"
    assert "You are helpful." in calls[1][0]["content"]
    assert "How are you" in calls[1][0]["content"]


def test_inference_service_chat_stream_trims_end_of_turn_markers(monkeypatch):
    """Chat streaming should stop before model-specific turn markers leak into the reply."""
    from app.schemas.inference import ChatMessage, Role
    from app.services.inference import InferenceService

    svc = InferenceService()
    svc._model = object()

    class FakeTokenizer:
        def encode(self, text):
            return text.split()

    svc._tokenizer = FakeTokenizer()

    monkeypatch.setattr(svc, "_apply_chat_template", lambda messages: "prompt")
    monkeypatch.setattr(
        svc,
        "generate_stream",
        lambda prompt, max_tokens=None, temperature=None, top_p=None, repetition_penalty=None: iter(
            [
                ("I am doing well", None),
                (", thank you<end_of_turn>", None),
                (
                    "",
                    {
                        "prompt_tokens": 5,
                        "completion_tokens": 4,
                        "finish_reason": "length",
                        "metrics": {"total_duration_s": 0.5},
                    },
                ),
            ]
        ),
    )

    chunks = list(svc.chat_stream(messages=[ChatMessage(role=Role.user, content="How are you")]))

    assert "".join(chunk for chunk, _usage in chunks) == "I am doing well, thank you"
    assert chunks[-1][1]["prompt_tokens"] == 1
    assert chunks[-1][1]["completion_tokens"] == 6
    assert chunks[-1][1]["finish_reason"] == "stop"
    assert chunks[-1][1]["metrics"]["total_duration_s"] is not None


def test_inference_service_generate_stream_synthesizes_usage_without_finish_reason(monkeypatch):
    """Verbose metrics should still be available when mlx_lm never emits a final finish marker."""
    import sys
    from types import SimpleNamespace

    from app.services.inference import InferenceService

    svc = InferenceService()
    svc._model = object()

    class FakeTokenizer:
        def encode(self, text):
            return text.split()

    svc._tokenizer = FakeTokenizer()

    fake_mlx_lm = SimpleNamespace(
        stream_generate=lambda model, tokenizer, prompt, **kwargs: iter(
            [
                SimpleNamespace(text="I am", finish_reason=None),
                SimpleNamespace(text=" well", finish_reason=None),
            ]
        )
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setattr(svc, "_generation_kwargs", lambda **kwargs: {})

    chunks = list(svc.generate_stream(prompt="How are you"))

    assert "".join(chunk for chunk, _usage in chunks) == "I am well"
    assert chunks[-1][1] is not None
    assert chunks[-1][1]["finish_reason"] == "stop"
    assert chunks[-1][1]["prompt_tokens"] == 3
    assert chunks[-1][1]["completion_tokens"] == 3
    assert chunks[-1][1]["metrics"]["total_duration_s"] is not None
