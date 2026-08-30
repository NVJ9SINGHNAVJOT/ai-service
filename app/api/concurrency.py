"""
Chat-request serialization for the HTTP layer.

The API is a single uvicorn process with one event loop, and MLX generation is
blocking C code. Two things follow, and this module supplies both:

**A gate.** `chat_lock()` is an ``asyncio.Lock`` held across load *and* the whole
generation, so only one chat request touches the inference singletons at a time.
Without it, an SSE stream yields between tokens, letting a second request unload
the model it is still generating from — the ``threading.RLock`` inside the
services cannot stop that, because both requests run on the same thread and the
lock is reentrant.

**A thread.** All chat work runs on one dedicated worker thread, off the event
loop. It must be *one* thread, not the shared pool: ``chat_stream`` holds the
service's ``RLock`` across the entire generator body, and an ``RLock`` may only
be released by the thread that acquired it. Starlette's ``run_in_threadpool`` /
``iterate_in_threadpool`` do not pin a task to a thread, so pumping a generator
through them would eventually release from the wrong one.

HTTP-only, like `middleware.py` and `response.py` — the CLI runs one chat loop
per process and needs none of this.
"""

from __future__ import annotations

import asyncio
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncGenerator, Callable, Iterator, Optional, TypeVar

T = TypeVar("T")

#: One lock per event loop. An ``asyncio.Lock`` binds to the loop that first
#: awaits it and rejects any other, so a module-level singleton would break the
#: moment a second TestClient built a fresh loop.
_locks: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = (
    weakref.WeakKeyDictionary()
)

#: Created on demand and rebuilt after a shutdown, so a restarted lifespan (or
#: the next TestClient in a test run) gets a working thread instead of a dead one.
_executor: Optional[ThreadPoolExecutor] = None
_executor_guard = threading.Lock()

#: Sentinel distinguishing "generator exhausted" from a legitimately yielded None.
_DONE = object()


def _chat_executor() -> ThreadPoolExecutor:
    """The single chat worker thread, started on first use."""
    global _executor
    with _executor_guard:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chat")
        return _executor


def chat_lock() -> asyncio.Lock:
    """Return the chat gate for the running event loop, creating it on first use."""
    loop = asyncio.get_running_loop()
    lock = _locks.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _locks[loop] = lock
    return lock


async def acquire_chat_gate(timeout: float) -> None:
    """
    Acquire the chat gate, waiting at most `timeout` seconds.

    Args:
        timeout: Seconds to wait, or ``0`` to take the gate only if it is free
            right now (what the model load/unload routes want).

    Raises:
        TimeoutError: if the gate did not free up in time. The caller decides
            what that means — 503 for a queued chat, 409 for a control endpoint.
    """
    lock = chat_lock()
    if timeout == 0:
        if lock.locked():
            raise TimeoutError("chat gate is held")
        # An uncontended acquire never awaits, so nothing can slip in between.
        await lock.acquire()
        return
    await asyncio.wait_for(lock.acquire(), timeout)


async def run_chat(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run one blocking chat call on the dedicated chat thread."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_chat_executor(), lambda: fn(*args, **kwargs))


async def aiter_chat(sync_iter: Iterator[T]) -> AsyncGenerator[T, None]:
    """
    Pump a blocking generator through the chat thread, one item per ``next()``.

    Closing is deliberate: on client disconnect the consumer stops iterating, and
    ``close()`` must run on the same thread so the ``GeneratorExit`` unwinds the
    generator's ``with self._lock`` where that lock is actually owned.
    """
    loop = asyncio.get_running_loop()
    executor = _chat_executor()

    def _next() -> Any:
        try:
            return next(sync_iter)
        except StopIteration:
            return _DONE

    try:
        while True:
            item = await loop.run_in_executor(executor, _next)
            if item is _DONE:
                return
            yield item
    finally:
        await loop.run_in_executor(executor, sync_iter.close)


def shutdown_chat_executor() -> None:
    """Wait for in-flight chat work and retire the worker thread (lifespan shutdown)."""
    global _executor
    with _executor_guard:
        executor, _executor = _executor, None
    if executor is not None:  # released the guard first — shutdown() blocks
        executor.shutdown(wait=True)
