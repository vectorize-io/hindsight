"""Unit tests for the sync->async loop bridge (hindsight_litellm._async)."""

import asyncio

import pytest

from hindsight_litellm import _async


@pytest.fixture(autouse=True)
def _restore_thread_loop():
    """Keep the bridge's per-thread loop state from leaking into other tests."""
    yield
    _async.close_loop()
    # Hand the thread a fresh, open loop so later (e.g. pytest-asyncio) tests
    # aren't surprised by a missing current loop.
    asyncio.set_event_loop(asyncio.new_event_loop())


def test_ensure_loop_reuses_same_loop():
    loop1 = _async.ensure_loop()
    loop2 = _async.ensure_loop()
    assert loop1 is loop2
    assert not loop1.is_closed()


def test_ensure_loop_sets_thread_current_loop():
    loop = _async.ensure_loop()
    # The client's internal get_event_loop() must resolve to our loop (no
    # "no current event loop" deprecation).
    assert asyncio.get_event_loop() is loop


def test_run_sync_runs_coroutine():
    async def _coro():
        return 21 * 2

    assert _async.run_sync(_coro()) == 42


def test_run_sync_reuses_loop_across_calls():
    async def _coro(n):
        return n

    _async.run_sync(_coro(1))
    loop_after_first = _async._local.loop
    _async.run_sync(_coro(2))
    assert _async._local.loop is loop_after_first


def test_close_loop_clears_and_recovers():
    loop = _async.ensure_loop()
    _async.close_loop()
    assert loop.is_closed()
    assert _async._local.loop is None
    # The next ensure_loop() must produce a fresh, open loop — never the closed
    # one — so callers transparently recover after a close.
    new_loop = _async.ensure_loop()
    assert new_loop is not loop
    assert not new_loop.is_closed()


def test_run_sync_recovers_after_close():
    async def _coro():
        return "ok"

    _async.run_sync(_coro())
    _async.close_loop()
    # Should transparently create a fresh loop and succeed.
    assert _async.run_sync(_coro()) == "ok"


def test_ensure_loop_noops_inside_running_loop():
    async def _inside():
        return _async.ensure_loop()

    assert asyncio.run(_inside()) is None
