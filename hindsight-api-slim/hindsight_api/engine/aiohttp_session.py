"""Shared aiohttp plumbing for every outbound HTTP call the API makes itself.

Production code talks HTTP through aiohttp only, and only asynchronously (see the
code-review skill). Two things every call site would otherwise re-derive:

* **A session per event loop.** An ``aiohttp.ClientSession`` (and its connector)
  binds to the loop that created it. A Hindsight process runs more than one loop —
  provider ``initialize()`` is sometimes run under ``asyncio.run`` in an executor
  thread, and tests and tooling spin up their own — so a session created eagerly in
  ``initialize()`` would be unusable from the serving loop. :class:`LoopLocalSession`
  creates one lazily, on first use, for whichever loop is running.
* **httpx-shaped timeouts.** httpx applied a bare float to each phase (connect, and
  every socket read) rather than to the whole exchange; aiohttp's ``total`` covers the
  entire request including the body. :func:`per_phase_timeout` keeps the old per-phase
  semantics so a long streamed response is not cut off by what used to be an idle timeout.
"""

from __future__ import annotations

import threading
import weakref
from asyncio import AbstractEventLoop, get_running_loop
from collections.abc import Callable, Mapping
from typing import Generic, TypeVar

import aiohttp

T = TypeVar("T")


def per_phase_timeout(seconds: float | None, *, connect: float | None = None) -> aiohttp.ClientTimeout:
    """Timeout applied per phase (pool wait + connect, and each socket read), not to the whole request.

    ``connect`` defaults to ``seconds``. ``None`` disables the corresponding bound.
    """
    return aiohttp.ClientTimeout(
        total=None,
        connect=seconds if connect is None else connect,
        sock_read=seconds,
    )


class UpstreamHTTPError(Exception):
    """A non-2xx response, with the body already read.

    ``aiohttp.ClientResponseError`` drops the body, which is usually the only place an
    upstream says *why* it refused. ``status_code`` (not aiohttp's ``status``) is the
    attribute :func:`remote_retry.status_code_of` classifies on.
    """

    def __init__(self, status_code: int, body: str, headers: Mapping[str, str], url: str) -> None:
        super().__init__(f"HTTP {status_code} from {url}: {body[:500]}")
        self.status_code = status_code
        self.body = body
        self.headers = headers
        self.url = url


async def raise_for_status(response: aiohttp.ClientResponse) -> None:
    """Raise :class:`UpstreamHTTPError` for a non-2xx response, carrying its body."""
    if response.status < 400:
        return
    try:
        body = await response.text()
    except (aiohttp.ClientError, UnicodeDecodeError):
        body = ""
    raise UpstreamHTTPError(response.status, body, response.headers, str(response.url))


class LoopLocal(Generic[T]):
    """One lazily-built ``T`` per running event loop.

    For state that binds to the loop that first uses it: an ``aiohttp.ClientSession``,
    an ``asyncio.Semaphore``, or an async SDK client whose pooled connections belong to
    the loop that opened them. Built on first use from inside a coroutine — never in
    ``__init__`` or ``initialize()``, which may run under ``asyncio.run`` in an executor
    thread.

    Entries for loops that have since closed are pruned on every :meth:`get`. A weak
    map would not do it: the stored value (a session, a lock) holds a strong reference
    to its own loop, so the key would never be collected and every throwaway
    ``asyncio.run`` loop would stay alive with its sockets.
    """

    def __init__(self, factory: Callable[[], T], *, discard: Callable[[T], None] | None = None) -> None:
        self._factory = factory
        self._discard = discard
        self._items: dict[AbstractEventLoop, T] = {}
        # Guards the map only (no await inside), so it is safe across loops and threads.
        self._lock = threading.Lock()

    def get(self) -> T:
        loop = get_running_loop()
        with self._lock:
            self._prune_closed_loops()
            item = self._items.get(loop)
            if item is None:
                item = self._factory()
                self._items[loop] = item
            return item

    def replace(self, item: T) -> None:
        """Swap in a new value for the running loop (e.g. after the old one was closed)."""
        with self._lock:
            self._items[get_running_loop()] = item

    def pop_running(self) -> T | None:
        """Remove and return the running loop's value, if it has one."""
        with self._lock:
            return self._items.pop(get_running_loop(), None)

    def _prune_closed_loops(self) -> None:
        for loop in [loop for loop in self._items if loop.is_closed()]:
            item = self._items.pop(loop)
            if self._discard is not None:
                self._discard(item)


def _discard_session_of_closed_loop(session: aiohttp.ClientSession) -> None:
    # ``ClientSession.close()`` is a coroutine and the loop it needs is gone, so release
    # the connector synchronously instead: ``BaseConnector._close`` is the sync half of
    # ``close()`` and, on a closed loop, only marks the pool closed. Without it the
    # dropped session logs "Unclosed client session" when collected.
    connector = session.connector
    session.detach()
    if connector is not None:
        connector._close()


class LoopLocalSession:
    """Lazily-created ``aiohttp.ClientSession``, one per running event loop.

    Construct it anywhere (no loop needed); call :meth:`get` from inside a coroutine.
    Every instance is registered so :func:`close_loop_sessions` can close them all at
    shutdown — providers and rerankers have no close hook of their own.
    """

    def __init__(
        self,
        *,
        timeout: aiohttp.ClientTimeout,
        headers: Mapping[str, str] | None = None,
        connector_factory: Callable[[], aiohttp.BaseConnector] | None = None,
    ) -> None:
        self._timeout = timeout
        self._headers = dict(headers) if headers else None
        self._connector_factory = connector_factory
        self._sessions: LoopLocal[aiohttp.ClientSession] = LoopLocal(
            self._new_session, discard=_discard_session_of_closed_loop
        )
        with _registry_lock:
            _registry.add(self)

    def _new_session(self) -> aiohttp.ClientSession:
        connector = self._connector_factory() if self._connector_factory else None
        return aiohttp.ClientSession(timeout=self._timeout, headers=self._headers, connector=connector)

    def get(self) -> aiohttp.ClientSession:
        session = self._sessions.get()
        if session.closed:
            session = self._new_session()
            self._sessions.replace(session)
        return session

    async def close(self) -> None:
        """Close the session owned by the running loop."""
        session = self._sessions.pop_running()
        if session is not None and not session.closed:
            await session.close()


_registry: weakref.WeakSet[LoopLocalSession] = weakref.WeakSet()
_registry_lock = threading.Lock()


async def close_loop_sessions() -> None:
    """Close every session the running loop opened, across all holders (engine shutdown, test teardown)."""
    with _registry_lock:
        holders = list(_registry)
    for holder in holders:
        await holder.close()
