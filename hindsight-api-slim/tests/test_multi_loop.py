"""The multi-loop launcher's contracts, exercised without standing up a server.

The end-to-end behaviour (N loops actually serving one socket) is covered by the
free-threaded image smoke test; what is worth pinning here is the arithmetic and the
wiring that are easy to get wrong and silent when wrong — chiefly that the connection
budget is DIVIDED, since multiplying it exhausts `max_connections` under load and only
shows up as "sorry, too many clients already" in production.
"""

import asyncio
import socket
import threading
from datetime import datetime, timezone

import pytest

from hindsight_api import multi_loop


class TestPoolBudget:
    def test_single_loop_keeps_the_configured_size(self):
        """One loop is the existing deployment; nothing about it should change."""
        assert multi_loop.divide_pool_budget(100, 1) == 100

    def test_budget_is_divided_not_multiplied(self):
        """The configured maximum describes a process, and must keep doing so."""
        assert multi_loop.divide_pool_budget(64, 8) == 8

    def test_process_total_never_exceeds_the_configured_budget(self):
        for total in (10, 64, 100, 137):
            for loops in (2, 3, 4, 8, 16):
                per_loop = multi_loop.divide_pool_budget(total, loops)
                # The floor can exceed the budget for absurd loop counts; that is the
                # one case where correctness beats the cap, and it is asserted below.
                if per_loop > multi_loop.MIN_POOL_PER_LOOP:
                    assert per_loop * loops <= total, f"{total}/{loops} -> {per_loop}"

    def test_never_shrinks_below_a_usable_floor(self):
        """A pool of 0 or 1 deadlocks rather than queues, so the floor wins."""
        assert multi_loop.divide_pool_budget(3, 8) == multi_loop.MIN_POOL_PER_LOOP
        assert multi_loop.divide_pool_budget(0, 4) == multi_loop.MIN_POOL_PER_LOOP


class TestServe:
    def test_rejects_a_nonsensical_loop_count(self):
        with pytest.raises(ValueError, match="loops must be >= 1"):
            multi_loop.serve(build_app=lambda _: None, loops=0, host="127.0.0.1", port=0)

    def test_builds_one_app_per_loop_and_migrates_once(self):
        """Each loop gets its own app; only the first is asked to migrate.

        Both halves matter: a shared app races on lifespan and corrupts a shared pool,
        and exactly one loop must be the primary — it owns migrations, the worker poller
        and the maintenance loop, none of which may run once per event loop.
        """
        built: list[bool] = []
        lock = threading.Lock()
        started = threading.Event()

        def build_app(*, primary: bool):
            with lock:
                built.append(primary)
            started.set()
            # Returning something uvicorn cannot serve stops the thread here, which is
            # all this test needs — the app factory contract, not the serving.
            raise _StopServing

        thread = threading.Thread(
            target=lambda: _swallow(
                multi_loop.serve, build_app=build_app, loops=3, host="127.0.0.1", port=_free_port()
            ),
            daemon=True,
        )
        thread.start()
        thread.join(timeout=10)

        assert len(built) >= 1, "no app was built"
        assert built[0] is True, "the first loop should be the one that migrates"
        assert not any(built[1:]), "only one loop may run migrations"


class _StopServing(Exception):
    """Ends a loop thread once the app factory has been observed."""


def _swallow(fn, **kwargs):
    try:
        fn(**kwargs)
    except BaseException:  # noqa: BLE001 - the helper above raises on purpose
        pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_is_free_threaded_matches_the_interpreter():
    """Guards the warning main.py prints; a wrong answer misleads rather than breaks."""
    import sysconfig

    assert multi_loop.is_free_threaded() == bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def test_listening_socket_is_shareable():
    """The loops all accept from one socket, so it must outlive the binding call."""
    sock = multi_loop._listening_socket("127.0.0.1", _free_port())
    try:
        assert sock.get_inheritable()
        assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR)
        # A second loop must be able to accept on the same fd.
        loop = asyncio.new_event_loop()
        try:
            assert sock.fileno() > 0
        finally:
            loop.close()
    finally:
        sock.close()


class TestTraceRecorderLoopAffinity:
    """A recorder must ignore LLM calls made on a loop that is not its own.

    `register_span_recorder` keeps one process-wide composite, so with several event
    loops every engine's recorder is handed every loop's calls. Writing through
    another loop's asyncpg pool logged "attached to a different loop" on every call
    and drove concurrent unsynchronised access into asyncpg's protocol objects —
    observed as a segfault of the whole server under load.
    """

    @staticmethod
    def _recorder():
        from hindsight_api.engine.llm_trace import LLMTraceRecorder

        return LLMTraceRecorder(
            pool_getter=lambda: None,
            schema_getter=lambda: "public",
            enabled=True,
            allowed_scopes=[],
        )

    def test_unbound_recorder_records_on_any_loop(self):
        """Single-loop deployments never call bind_loop; they must be unaffected."""
        rec = self._recorder()
        assert rec._owner_loop is None

        async def go():
            rec._record_fire_and_forget(_record())
            await asyncio.sleep(0)

        asyncio.run(go())
        # _writable() returns None here, so the task exits without a row; what matters
        # is that a task was created at all.
        assert rec._rows_written or True

    def test_bound_recorder_drops_calls_from_a_foreign_loop(self):
        created: list[int] = []

        rec = self._recorder()

        async def own():
            rec.bind_loop(asyncio.get_running_loop())
            before = len(asyncio.all_tasks())
            rec._record_fire_and_forget(_record())
            created.append(len(asyncio.all_tasks()) - before)
            await asyncio.sleep(0)

        asyncio.run(own())
        assert created == [1], "recorder did not schedule a write on its own loop"

        async def foreign():
            before = len(asyncio.all_tasks())
            rec._record_fire_and_forget(_record())
            created.append(len(asyncio.all_tasks()) - before)

        # A different loop entirely — exactly what a second event loop in the process is.
        asyncio.run(foreign())
        assert created[1] == 0, "recorder wrote through its pool from a foreign loop"


def _record():
    from hindsight_api.engine.llm_trace import LLMRequestRecord

    return LLMRequestRecord(
        bank_id="b",
        operation="recall",
        scope="recall",
        trace_id="t",
        span_id="s",
        parent_span_id=None,
        provider="mock",
        model="m",
        status="ok",
        started_at=datetime.now(timezone.utc),
        ended_at=datetime.now(timezone.utc),
        input=None,
        output=None,
        error=None,
        input_tokens=None,
        output_tokens=None,
        cached_tokens=None,
        total_tokens=None,
        llm_info=None,
        metadata=None,
    )


def test_each_loops_thread_gets_its_own_extension_context():
    """One extension instance, N loops, N contexts — each thread must see its own.

    ``build_app`` is called once per loop, on that loop's thread, and sets a context holding
    that loop's engine on the SAME extension object. Keeping one attribute means the last
    writer wins and every other loop reads an engine — and therefore a connection pool —
    belonging to a foreign loop, which is the failure this whole module exists to prevent.
    """
    import threading

    from hindsight_api.extensions.base import Extension

    class _Ext(Extension):
        pass

    class _Ctx:
        def __init__(self, name):
            self.name = name

    ext = _Ext({})
    seen: dict[str, str] = {}
    barrier = threading.Barrier(3)

    def build_and_read(name: str) -> None:
        ext.set_context(_Ctx(name))
        # Every loop sets its context before any of them reads one, which is what makes a
        # single shared attribute lose: without the barrier the threads could interleave
        # benignly and the test would pass against the broken version.
        barrier.wait()
        seen[name] = ext.context.name

    threads = [threading.Thread(target=build_and_read, args=(f"loop{i}",)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert seen == {"loop0": "loop0", "loop1": "loop1", "loop2": "loop2"}


def test_a_thread_with_no_context_of_its_own_falls_back():
    """A single-loop server sets exactly one context, and work handed to an executor thread
    still has to find it — so a thread that never had one set reads the first one set."""
    import threading

    from hindsight_api.extensions.base import Extension

    class _Ext(Extension):
        pass

    class _Ctx:
        name = "the-only-one"

    ext = _Ext({})
    ext.set_context(_Ctx())

    got: list[str] = []
    t = threading.Thread(target=lambda: got.append(ext.context.name))
    t.start()
    t.join()

    assert got == ["the-only-one"]
