"""
Tests for the interactive model picker (`app/cli/select.py`) and the
`cli` command that wires it to a chat session.
"""

from __future__ import annotations

from typer.testing import CliRunner

from app.schemas.model import ModelInfo, ModelSource, ModelState


def _model(name: str, *, loadable: bool = True, state: ModelState = ModelState.ready,
           modalities=("text",)) -> ModelInfo:
    return ModelInfo(
        name=name,
        source=ModelSource.downloaded,
        path=f"/tmp/{name}",
        loadable=loadable,
        state=state,
        input_modalities=list(modalities),
    )


# ── Selector fallback (non-TTY path) ─────────────────────────────────────────

def test_select_fallback_returns_index(monkeypatch):
    from app.cli import select

    monkeypatch.setattr(select.Prompt, "ask", lambda *a, **k: "2")
    assert select._select_fallback([("a",), ("b",), ("c",)], "Pick") == 1


def test_select_fallback_invalid_returns_none(monkeypatch):
    from app.cli import select

    monkeypatch.setattr(select.Prompt, "ask", lambda *a, **k: "nope")
    assert select._select_fallback([("a",)], "Pick") is None


def test_select_fallback_out_of_range_returns_none(monkeypatch):
    from app.cli import select

    monkeypatch.setattr(select.Prompt, "ask", lambda *a, **k: "9")
    assert select._select_fallback([("a",), ("b",)], "Pick") is None


def test_select_from_list_empty_returns_none():
    from app.cli import select

    assert select.select_from_list([], "Pick") is None


# ── `cli` command wiring ─────────────────────────────────────────────────────

def test_cli_command_offers_only_loadable_and_runs_chat(monkeypatch):
    import app.cli.main as main
    import app.cli.select as select
    import app.services.model_manager as mm

    models = [
        _model("good-1"),
        _model("good-2", modalities=("text", "image")),
        _model("bad", loadable=False, state=ModelState.incomplete),
    ]

    class FakeManager:
        def list_models(self):
            return models

    monkeypatch.setattr(mm, "ModelManager", lambda *a, **k: FakeManager())

    captured: dict = {}

    def fake_select(rows, title, columns=None):
        captured["rows"] = list(rows)
        return 0

    monkeypatch.setattr(select, "select_from_list", fake_select)

    called: dict = {}
    monkeypatch.setattr(main, "_run_chat", lambda **kwargs: called.update(kwargs))

    result = CliRunner().invoke(main.cli, ["cli", "--verbose"])

    assert result.exit_code == 0
    # Only the two loadable models were offered, in order.
    assert [row[0] for row in captured["rows"]] == ["good-1", "good-2"]
    # The selected model name was forwarded to the chat dispatch with flags.
    assert called["model"] == "good-1"
    assert called["verbose"] is True


def test_cli_command_cancelled_does_not_chat(monkeypatch):
    import app.cli.main as main
    import app.cli.select as select
    import app.services.model_manager as mm

    class FakeManager:
        def list_models(self):
            return [_model("good-1")]

    monkeypatch.setattr(mm, "ModelManager", lambda *a, **k: FakeManager())
    monkeypatch.setattr(select, "select_from_list", lambda *a, **k: None)

    ran: dict = {"called": False}
    monkeypatch.setattr(main, "_run_chat", lambda **kwargs: ran.update(called=True))

    result = CliRunner().invoke(main.cli, ["cli"])

    assert result.exit_code == 0
    assert ran["called"] is False
    assert "Cancelled" in result.stdout


def test_cli_command_no_loadable_models(monkeypatch):
    import app.cli.main as main
    import app.cli.select as select
    import app.services.model_manager as mm

    class FakeManager:
        def list_models(self):
            return [_model("bad", loadable=False, state=ModelState.incomplete)]

    monkeypatch.setattr(mm, "ModelManager", lambda *a, **k: FakeManager())

    def _should_not_run(*a, **k):
        raise AssertionError("selector should not be shown when nothing is loadable")

    monkeypatch.setattr(select, "select_from_list", _should_not_run)

    ran: dict = {"called": False}
    monkeypatch.setattr(main, "_run_chat", lambda **kwargs: ran.update(called=True))

    result = CliRunner().invoke(main.cli, ["cli"])

    assert result.exit_code == 0
    assert ran["called"] is False
    assert "No loadable models found" in result.stdout
