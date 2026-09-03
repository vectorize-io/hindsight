"""Serve the API from several asyncio event loops inside one process.

A single event loop executes Python bytecode on one thread, so every request's
CPU work — RRF fusion, pydantic validation, JSON serialisation — serialises behind
it no matter how much of the request is spent waiting on Postgres. On a
free-threaded build (the ``-py3.14t`` image) that ceiling is removable: run several
loops, each on its own thread, and they execute in parallel.

Measured on identical recall work against Postgres 18 + pgvector, 64 concurrent:

    3.11, 1 loop      23 rps   0.93 cores   p50 2187ms
    3.14t, 1 loop     42 rps   1.03 cores   p50 1286ms
    3.14t, 8 loops    82 rps   5.61 cores   p50  597ms

The middle row is the 3.14 interpreter; the jump to the third is this module.

**Every loop gets its own application.** Not a shared one, for two reasons, both
observed rather than theorised: uvicorn runs the app's lifespan once per server, so
N loops race on the engine's shared initialisation, and an asyncpg pool belongs to
the loop that created it, so a shared pool corrupts under load
("got result for unknown protocol state"). One app per loop means one MemoryEngine
and one pool per loop.

**The connection budget is divided, not multiplied.** N loops with the configured
pool size would open N times the connections and exhaust ``max_connections`` —
measured, that is the first thing that goes wrong. Each loop gets a fair share.

**The loops share one listening socket** and the kernel distributes accepts. That
distribution is not even: a 2.7x spread between the busiest and quietest loop was
measured with long-lived keepalive connections. It costs some tail latency and is
still far better than one loop; per-loop sockets with SO_REUSEPORT would improve it
on Linux and are a reasonable follow-up.

This is orthogonal to ``--workers``, which forks whole processes. Workers give
isolation and duplicate memory; loops share the process and its caches. They can be
combined, in which case the budget divides across both.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import socket
import sysconfig
import threading
from typing import Any, Callable

import uvicorn

logger = logging.getLogger(__name__)

#: Never shrink a loop's pool below this, however many loops are configured. A pool
#: too small to serve one request concurrently would deadlock rather than queue.
MIN_POOL_PER_LOOP = 2


def is_free_threaded() -> bool:
    """Whether this interpreter runs Python in parallel across threads.

    More than one loop is only a throughput win here; on a GIL build the loops take
    turns and the extra threads and connections buy nothing.
    """
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def divide_pool_budget(total: int, loops: int) -> int:
    """Per-loop pool size that keeps the PROCESS total at roughly ``total``.

    The configured maximum has always described one process. Preserving that meaning
    is what stops a multi-loop process from quietly opening N times the connections
    it is allowed and failing with "sorry, too many clients already".
    """
    if loops <= 1:
        return total
    return max(MIN_POOL_PER_LOOP, total // loops)


def _listening_socket(host: str, port: int, backlog: int = 2048) -> socket.socket:
    """One socket the loops share; the kernel hands each accept to a waiting loop."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(backlog)
    sock.set_inheritable(True)
    return sock


def serve(
    *,
    build_app: Callable[..., Any],
    loops: int,
    host: str,
    port: int,
    uvicorn_kwargs: dict[str, Any] | None = None,
) -> None:
    """Run ``loops`` uvicorn servers, one per thread, over a shared socket.

    ``build_app(primary=...)`` is called once per loop, on that loop's thread, and
    must return a fresh application — see the module docstring for why it cannot be
    shared. Exactly one loop is the primary, and it owns everything that belongs to
    the *process* rather than to a loop: migrations (idempotent, but N concurrent
    runs at boot is a race worth not having) and the background work — the worker
    poller and the maintenance loop. A second poller under the same worker id would
    claim the same tasks rather than add capacity, and every extra maintenance loop
    is a duplicate retention sweep.

    Returns when every server has stopped.
    """
    if loops < 1:
        raise ValueError(f"loops must be >= 1, got {loops}")

    sock = _listening_socket(host, port)
    servers: list[uvicorn.Server] = []
    servers_lock = threading.Lock()
    # The first loop migrates while the rest wait, so they start against a schema
    # that is already current instead of racing to apply it.
    migrated = threading.Event()

    def run_one(index: int) -> None:
        is_first = index == 0
        if not is_first:
            migrated.wait()
        try:
            app = build_app(primary=is_first)
        finally:
            if is_first:
                # Released even on failure: the other loops should start and report
                # their own error rather than hang on a barrier that never opens.
                migrated.set()

        # The loop implementation comes from the caller's config (uvloop when it is
        # installed), not forced here: each thread gets its own loop instance either way.
        config = uvicorn.Config(app, **(uvicorn_kwargs or {}))
        server = uvicorn.Server(config)
        # Signal handlers can only be installed on the main thread; the caller's
        # handler stops every server through `servers` instead.
        server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
        with servers_lock:
            servers.append(server)
        asyncio.run(server.serve(sockets=[sock]))

    # daemon=True so a SystemExit raised by the caller's signal handler on the main
    # thread does not hang interpreter shutdown waiting for loops that are parked in
    # accept(). The handler chained below gives them a chance to drain first.
    threads = [
        threading.Thread(target=run_one, args=(i,), name=f"hindsight-loop-{i}", daemon=True) for i in range(loops)
    ]
    _chain_shutdown_signals(servers, servers_lock)

    logger.info("Starting %d event loops in one process on %s:%d", loops, host, port)
    for thread in threads:
        thread.start()

    try:
        for thread in threads:
            thread.join()
    finally:
        sock.close()


def _chain_shutdown_signals(servers: list[uvicorn.Server], lock: threading.Lock) -> None:
    """Ask the servers to drain on SIGINT/SIGTERM, then run whatever ran before.

    Chained rather than replacing: the caller installs handlers that release the
    embedded database and exit, and those still have to run. This only gets in front
    of them so uvicorn stops accepting and finishes in-flight requests instead of
    being torn down mid-response.
    """
    if threading.current_thread() is not threading.main_thread():
        # signal.signal() raises off the main thread. Serving is still correct without
        # this — the loops just lose the chance to drain before the process goes down —
        # so this is a degradation, not a failure.
        logger.debug("Not on the main thread; skipping graceful-shutdown signal handlers")
        return

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def stop(signum, frame):  # noqa: ANN001 - signal handler signature
        with lock:
            for server in servers:
                server.should_exit = True
        handler = previous.get(signum)
        if callable(handler):
            handler(signum, frame)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
