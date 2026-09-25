"""LoopLocalSession / raise_for_status: the shared aiohttp plumbing every client builds on."""

import ast
import asyncio
import pathlib
import threading

import aiohttp
import pytest
from aiohttp import web

import hindsight_api
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


@pytest.mark.asyncio
async def test_request_goes_through_the_proxy_named_by_http_proxy(monkeypatch):
    # The httpx -> aiohttp migration (#4318) dropped proxy support: httpx reads the
    # proxy env vars by default, aiohttp only with trust_env. Prove it end to end by
    # pointing HTTP_PROXY at a stub and asking for a host that does not resolve —
    # the stub is reached only if the request was actually proxied.
    proxied: list[str] = []

    async def handler(request: web.Request) -> web.StreamResponse:
        proxied.append(str(request.url))
        return web.Response(text="via proxy")

    async with stub_server(handler) as proxy_url:
        monkeypatch.setenv("HTTP_PROXY", proxy_url)
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)
        holder = LoopLocalSession(timeout=per_phase_timeout(5.0))
        async with holder.get().get("http://unreachable.invalid/embeddings") as response:
            assert await response.text() == "via proxy"
        await holder.close()

    assert proxied == ["http://unreachable.invalid/embeddings"]


def test_no_proxy_env_means_no_per_request_env_lookup(monkeypatch):
    # trust_env costs two executor hops per request (netrc + getproxies), so it stays
    # off unless the deployment actually set a proxy.
    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    holder = LoopLocalSession(timeout=per_phase_timeout(5.0))

    async def check() -> bool:
        session = holder.get()
        trust_env = session.trust_env
        await holder.close()
        return trust_env

    assert asyncio.run(check()) is False


def test_every_production_session_decides_about_the_proxy():
    """Every server-side ``aiohttp.ClientSession`` says whether it follows the proxy env vars.

    Server-side is the API package and the extensions it loads; the SDK clients and the
    LiteLLM callback run in someone else's process and are not covered here.

    A new client that forgets is the failure this guards: it silently gets aiohttp's
    ``trust_env=False`` and bypasses the proxy, which is exactly how the migration
    regressed. Sessions built through ``LoopLocalSession`` inherit the decision, so
    only direct constructions are checked here.
    """
    package = pathlib.Path(hindsight_api.__file__).parent
    repo = package.parents[1]
    # The server package plus the extensions, which is where the supabase session that
    # reached an external endpoint without following the proxy was found.
    roots = [package, repo / "hindsight-extensions"]
    # aiohttp_session.py IS the shared decision, so it has nothing to defer to.
    exempt = {package / "engine" / "aiohttp_session.py"}
    missing = []
    seen = 0
    for root in roots:
        # The extensions live beside the package in the repo, but not in an installed
        # copy of it; skipping a root that isn't there is why `seen` is asserted below.
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            # Each extension carries its own .venv; only their own source counts.
            if path in exempt or any(part in {"tests", ".venv", "build"} for part in path.parts):
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                target = ast.unparse(node.func)
                if target not in ("aiohttp.ClientSession", "ClientSession"):
                    continue
                seen += 1
                if not any(kw.arg == "trust_env" for kw in node.keywords):
                    missing.append(f"{path.relative_to(repo)}:{node.lineno} {target}")
    assert missing == [], "sessions with no explicit trust_env decision: " + ", ".join(missing)
    assert seen >= 4, f"only {seen} sessions found — the scan is looking in the wrong place"
