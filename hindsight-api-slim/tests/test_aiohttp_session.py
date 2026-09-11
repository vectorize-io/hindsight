"""LoopLocalSession / raise_for_status: the shared aiohttp plumbing every client builds on."""

import asyncio
import threading

import aiohttp
import pytest
from aiohttp import web

from hindsight_api.engine.aiohttp_session import (
    LoopLocalSession,
    UpstreamHTTPError,
    close_loop_sessions,
    per_phase_timeout,
    raise_for_status,
)
from hindsight_api.engine.remote_retry import is_transient_remote_error
from tests.aiohttp_stub import stub_server


def test_constructed_without_a_running_loop():
    # Providers build their client in __init__, before any loop is running.
    LoopLocalSession(timeout=per_phase_timeout(5.0))


@pytest.mark.asyncio
async def test_reuses_one_session_per_loop():
    holder = LoopLocalSession(timeout=per_phase_timeout(5.0))
    first = holder.get()
    assert holder.get() is first
    await holder.close()
    assert first.closed
    # A closed session is replaced, not handed back.
    second = holder.get()
    assert second is not first and not second.closed
    await holder.close()


@pytest.mark.asyncio
async def test_other_loop_gets_its_own_session():
    """initialize() may run under asyncio.run in an executor thread; the serving loop must not inherit that session."""
    holder = LoopLocalSession(timeout=per_phase_timeout(5.0))
    other: list[aiohttp.ClientSession] = []

    async def use_on_other_loop() -> None:
        other.append(holder.get())
        await other[0].close()

    thread = threading.Thread(target=lambda: asyncio.run(use_on_other_loop()))
    thread.start()
    thread.join()

    mine = holder.get()
    assert mine is not other[0]
    assert not mine.closed
    await holder.close()


@pytest.mark.asyncio
async def test_close_loop_sessions_closes_every_holder():
    """Engine shutdown closes what providers opened; they have no close hook of their own."""
    holders = [LoopLocalSession(timeout=per_phase_timeout(5.0)) for _ in range(3)]
    sessions = [holder.get() for holder in holders]
    await close_loop_sessions()
    assert all(session.closed for session in sessions)


def test_closed_loops_are_released():
    """A throwaway asyncio.run loop (provider initialize() in an executor) must not be kept alive."""
    holder = LoopLocalSession(timeout=per_phase_timeout(5.0))
    opened: list[aiohttp.ClientSession] = []

    async def open_session() -> None:
        opened.append(holder.get())

    for _ in range(3):
        asyncio.run(open_session())

    async def after() -> None:
        holder.get()
        # Only the running loop's session is left; the dead loops' sessions are released.
        assert len(holder._sessions._items) == 1
        await holder.close()

    asyncio.run(after())
    assert all(session.closed for session in opened)


def test_per_phase_timeout_has_no_total_cap():
    timeout = per_phase_timeout(30.0, connect=5.0)
    assert timeout.total is None
    assert timeout.connect == 5.0
    assert timeout.sock_read == 30.0


@pytest.mark.asyncio
async def test_raise_for_status_keeps_body_and_is_classified():
    async def handler(request: web.Request) -> web.StreamResponse:
        status = int(request.query["status"])
        return web.Response(status=status, text=f"upstream says {status}", headers={"Retry-After": "2"})

    holder = LoopLocalSession(timeout=per_phase_timeout(5.0))
    async with stub_server(handler) as base_url:
        async with holder.get().get(f"{base_url}/x", params={"status": "200"}) as ok:
            await raise_for_status(ok)

        for status, transient in ((503, True), (429, True), (401, False)):
            async with holder.get().get(f"{base_url}/x", params={"status": str(status)}) as response:
                with pytest.raises(UpstreamHTTPError) as caught:
                    await raise_for_status(response)
            assert caught.value.status_code == status
            assert caught.value.body == f"upstream says {status}"
            assert caught.value.headers["Retry-After"] == "2"
            assert is_transient_remote_error(caught.value) is transient
    await holder.close()


def test_aiohttp_connection_errors_are_transient():
    assert is_transient_remote_error(aiohttp.ClientConnectionError("reset"))
    assert is_transient_remote_error(aiohttp.ServerTimeoutError("read timed out"))
