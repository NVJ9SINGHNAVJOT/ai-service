"""
Concurrency tests: chat serializes, audio runs in parallel per model.

Two independent gates are under test here:

- **Chat** — one request at a time across load and generation, streaming or not.
  A second chat waits (503 on timeout); a model load/unload is refused with 409.
- **Audio** — requests for the *same* model run concurrently; a request for a
  *different* model waits for the in-flight ones to drain before swapping.

Everything uses fakes and threads; no model is ever downloaded or loaded.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.schemas.model import ModelInfo, ModelSource


@pytest.fixture()
def shared_loop_client():
    """
    A TestClient entered as a context manager, so every request shares one loop.

    This matters here and nowhere else in the suite: the chat gate is an
    `asyncio.Lock` keyed by the running event loop, and an un-entered TestClient
    spins up a *fresh* loop per request — which would hand each request its own
    lock and quietly defeat the thing these tests exist to check. Under uvicorn
    there is only ever one loop, which is the case being reproduced.
    """
    from app.main import app

    with TestClient(app) as client:
        yield client


def _fake_model_info(name: str) -> ModelInfo:
    return ModelInfo(
        name=name,
        repo_id=None,
        source=ModelSource.custom,
        path="/tmp/fake-model",
        loadable=True,
        size_mb=None,
        created_at=None,
        updated_at=None,
    )


@pytest.fixture()
def blocking_chat(monkeypatch):
    """
    Stub the text backend so a streaming turn blocks until the test releases it.

    Yields a handle exposing:
      `started`  — set once generation has begun,
      `release`  — event the test sets to let generation finish,
      `loads`    — model names passed to load(), in order.
    """
    from app.api import routes_openai
    from app.main import inference_service
    from app.services.inference import InferenceService

    class Handle:
        def __init__(self) -> None:
            self.started = threading.Event()
            self.release = threading.Event()
            self.loads: list[str] = []
            self.loaded_name: str | None = None

    handle = Handle()

    from app.api import routes_models

    for manager in (routes_openai._manager, routes_models._manager):
        monkeypatch.setattr(manager, "ensure_model_loadable", _fake_model_info)
        monkeypatch.setattr(manager, "ensure_model_files_ready", _fake_model_info)

    def fake_load(model_path, model_name):
        handle.loads.append(model_name)
        handle.loaded_name = model_name

    monkeypatch.setattr(inference_service, "load", fake_load)
    monkeypatch.setattr(
        InferenceService, "loaded_model_name", property(lambda self: handle.loaded_name)
    )

    def fake_chat_stream(messages, **kwargs):
        # Yield repeatedly rather than blocking in one `next()`: each yield hands
        # the chat thread back, opening exactly the window a queued load would
        # slip through. Without that, the single-worker executor alone would
        # serialize the test and the gate would look effective when it isn't.
        handle.started.set()
        # Bounded (~10s): a failing assertion can leave `release` unset, and an
        # endless stream would hang the run instead of reporting the failure.
        for _ in range(500):
            if handle.release.wait(timeout=0.02):
                break
            yield "tick", None
        yield "", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

    monkeypatch.setattr(inference_service, "chat_stream", fake_chat_stream)
    monkeypatch.setattr(
        inference_service,
        "chat",
        lambda messages, **kwargs: ("buffered", {}),
    )
    try:
        yield handle
    finally:
        handle.release.set()  # never strand a stream on a failed assertion


def _stream_request(client: TestClient, model: str) -> int:
    """Drive a streaming completion to completion; return the status code."""
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": model, "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    ) as resp:
        for _ in resp.iter_lines():
            pass
        return resp.status_code


# ── Chat: one at a time ──────────────────────────────────────────────────────


def test_second_chat_waits_for_the_stream_to_finish(shared_loop_client, blocking_chat):
    """A chat for another model must not load until the in-flight stream ends."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        streaming = pool.submit(_stream_request, shared_loop_client, "model-a")
        assert blocking_chat.started.wait(timeout=5), "stream never started"

        second = pool.submit(
            shared_loop_client.post,
            "/v1/chat/completions",
            json={"model": "model-b", "messages": [{"role": "user", "content": "hi"}]},
        )

        # model-a keeps yielding for this whole window (~15 tokens), so the
        # queued request has many chances to interleave. It must take none.
        threading.Event().wait(0.3)
        assert blocking_chat.loads == ["model-a"], (
            f"model-b loaded mid-stream: {blocking_chat.loads}"
        )

        blocking_chat.release.set()
        assert streaming.result(timeout=10) == 200
        assert second.result(timeout=10).status_code == 200

    # Only after the stream drained did the swap happen.
    assert blocking_chat.loads == ["model-a", "model-b"]


def test_queued_chat_times_out_with_503(shared_loop_client, blocking_chat, monkeypatch):
    """A chat that can't get the gate within the timeout returns 503, not a hang."""
    from app.config import settings

    monkeypatch.setattr(settings, "chat_queue_timeout_seconds", 0.1)

    with ThreadPoolExecutor(max_workers=2) as pool:
        streaming = pool.submit(_stream_request, shared_loop_client, "model-a")
        assert blocking_chat.started.wait(timeout=5)

        resp = shared_loop_client.post(
            "/v1/chat/completions",
            json={"model": "model-b", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 503
        assert "busy" in resp.json()["detail"].lower()

        blocking_chat.release.set()
        assert streaming.result(timeout=10) == 200


def test_model_load_is_refused_with_409_while_generating(shared_loop_client, blocking_chat):
    """The control endpoints report 409 rather than yanking a model mid-stream."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        streaming = pool.submit(_stream_request, shared_loop_client, "model-a")
        assert blocking_chat.started.wait(timeout=5)

        load = shared_loop_client.post("/api/v1/models/load", json={"name": "model-b"})
        unload = shared_loop_client.post("/api/v1/models/unload", json={})
        assert load.status_code == 409
        assert unload.status_code == 409

        blocking_chat.release.set()
        assert streaming.result(timeout=10) == 200

    # The gate is free again, so control operations work.
    assert shared_loop_client.post("/api/v1/models/unload", json={}).status_code == 200


def test_disconnect_mid_stream_releases_the_gate(shared_loop_client, blocking_chat):
    """Abandoning a stream must not strand the chat gate or the service RLock."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        blocking_chat.release.set()  # let generation run freely

        def abort_early() -> None:
            with shared_loop_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "model-a",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            ) as resp:
                next(resp.iter_lines())  # read one frame, then walk away

        pool.submit(abort_early).result(timeout=10)

    # A follow-up request proves the gate was returned.
    resp = shared_loop_client.post(
        "/v1/chat/completions",
        json={"model": "model-a", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200


# ── Audio: parallel per model, serialized across models ──────────────────────


@pytest.fixture()
def audio_slot():
    """A `_ResidentModel` slot with instrumented load/release, no idle timer."""
    from app.services.audio import _ResidentModel

    class Recorder:
        def __init__(self) -> None:
            self.loads: list[str] = []
            self.evicted: list[str] = []

    rec = Recorder()
    slot = _ResidentModel("STT", lambda: 0.0, lambda: _record_eviction())

    def _record_eviction() -> None:
        # The hook also fires when the slot is empty (a no-op release on first
        # acquire), so only count evictions of a model that was actually there.
        if slot.loaded_name is not None:
            rec.evicted.append(slot.loaded_name)

    def loader(name: str):
        rec.loads.append(name)
        return object(), None

    return slot, rec, loader


def test_same_audio_model_runs_in_parallel(audio_slot):
    """Two requests for the resident model overlap and load it only once."""
    slot, rec, loader = audio_slot
    both_inside = threading.Barrier(2, timeout=5)

    def request() -> None:
        with slot.acquire("whisper", lambda: loader("whisper")):
            both_inside.wait()  # deadlocks (BrokenBarrier) unless truly concurrent

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(request) for _ in range(2)]
        for f in futures:
            f.result(timeout=10)

    assert rec.loads == ["whisper"], "the model should have loaded exactly once"


def test_different_audio_model_waits_for_the_drain(audio_slot):
    """A swap blocks until the in-flight request on the current model finishes."""
    slot, rec, loader = audio_slot
    first_inside = threading.Event()
    let_first_finish = threading.Event()
    order: list[str] = []

    def first() -> None:
        with slot.acquire("whisper", lambda: loader("whisper")):
            first_inside.set()
            let_first_finish.wait(timeout=5)
            order.append("first-done")

    def second() -> None:
        with slot.acquire("parakeet", lambda: loader("parakeet")):
            order.append("second-loaded")

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(first)
        assert first_inside.wait(timeout=5)
        f2 = pool.submit(second)

        # The swap must not have happened while the first request is inside.
        threading.Event().wait(0.2)
        assert rec.loads == ["whisper"], f"parakeet loaded mid-request: {rec.loads}"
        assert rec.evicted == []

        let_first_finish.set()
        f1.result(timeout=10)
        f2.result(timeout=10)

    assert order == ["first-done", "second-loaded"]
    assert rec.loads == ["whisper", "parakeet"]
    assert slot.loaded_name == "parakeet"
