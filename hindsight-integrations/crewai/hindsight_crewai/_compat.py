"""Async compatibility helpers.

CrewAI runs inside an async event loop. The Hindsight client's sync
methods internally call ``loop.run_until_complete()``, which fails
when a loop is already running or when ``asyncio.get_event_loop()``
returns a foreign loop from another thread.

This module provides a dedicated worker thread with a persistent
event loop for all Hindsight API calls, ensuring the aiohttp session
stays bound to a single, stable loop.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import Any, Callable

_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
_thread_init_lock = threading.Lock()
_initialized_threads: set[int] = set()


def _ensure_thread_loop() -> None:
    """Ensure the current thread has a persistent event loop."""
    tid = threading.get_ident()
    if tid not in _initialized_threads:
        with _thread_init_lock:
            if tid not in _initialized_threads:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                _initialized_threads.add(tid)


def call_sync(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a sync Hindsight client method safely.

    Runs the call in a dedicated thread pool where each thread has
    its own persistent event loop. This avoids:
    - Nested ``run_until_complete`` when CrewAI's loop is running
    - Cross-thread loop references that cause "Event loop is closed"
    - aiohttp session/loop binding issues
    """

    def _run() -> Any:
        _ensure_thread_loop()
        return fn(*args, **kwargs)

    future = _thread_pool.submit(_run)
    return future.result(timeout=60)
