"""Client lifecycle lock (#11923 in the original repo) — ``_client`` is read and
written from the retain writer thread, the background prefetch worker and the
turn/tool thread. ``_get_client()`` was check-then-act and the embedded
constructor takes seconds (runtime check, daemon spawn): two threads hitting a
cold or just-nulled client both construct, one wins, and the loser's client is
orphaned with an aiohttp session nothing ever closes. The stale-daemon retry
path (``self._client = None`` then ``_get_client()``) widened the window from
microseconds to seconds, and concurrent retries clobbered each other's
replacement. NousResearch/hermes-agent#117236."""

import threading
import time
import types


def test_client_created_once_under_concurrent_first_access(provider, monkeypatch):
    """8 threads against a cold cache and a slow factory: exactly one
    construction, one shared object."""
    instance, _ = provider({})
    instance._client = None
    built = []

    def _slow_build():
        time.sleep(0.2)  # force the check-then-act window open
        client = types.SimpleNamespace(name=f"client-{len(built)}")
        built.append(client)
        return client

    monkeypatch.setattr(instance, "_new_cloud_client", _slow_build)

    results = []

    def _fetch():
        results.append(instance._get_client())

    threads = [threading.Thread(target=_fetch, daemon=True) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert len(built) == 1
    assert len(results) == 8
    assert all(client is built[0] for client in results)


def _op_for(broken):
    def op(client):
        async def _attempt():
            if client is broken:
                raise RuntimeError("Cannot connect to host 127.0.0.1:8888")
            return "ok"

        return _attempt()

    return op


def test_retry_does_not_orphan_a_sibling_client(provider, monkeypatch):
    """The stale-daemon retry retires the client while sibling threads sit in
    ``_get_client()``; without the lock both build and the overwritten client is
    an orphan nobody closes. Exactly one replacement must ever be constructed."""
    instance, _ = provider({})
    instance._mode = "local_embedded"
    built = []

    def _build():
        time.sleep(0.2)  # widen the retire-and-rebuild window
        client = types.SimpleNamespace()
        built.append(client)
        return client

    monkeypatch.setattr(instance, "_new_embedded_client", _build)
    broken = _build()
    instance._client = broken

    stop = threading.Event()

    def _reader():
        while not stop.is_set():
            instance._get_client()

    readers = [threading.Thread(target=_reader, daemon=True) for _ in range(4)]
    for t in readers:
        t.start()
    try:
        assert instance._run_hindsight_operation(_op_for(broken)) == "ok"
    finally:
        stop.set()
        for t in readers:
            t.join(timeout=5.0)

    assert instance._client is not broken
    orphans = [c for c in built if c is not broken and c is not instance._client]
    assert orphans == []
    assert sum(c is not broken for c in built) == 1


def test_concurrent_retries_retire_the_broken_client_exactly_once(provider, monkeypatch):
    """Concurrent stale-daemon retries each run with the same broken client; the
    identity-argument retire (``_get_client(retire=client)``) must let exactly one
    of them rebuild it — never a shared broken-client flag a sibling could clobber,
    never two replacements orphaning each other."""
    instance, _ = provider({})
    instance._mode = "local_embedded"
    built = []

    def _build():
        time.sleep(0.2)
        client = types.SimpleNamespace()
        built.append(client)
        return client

    monkeypatch.setattr(instance, "_new_embedded_client", _build)
    broken = _build()
    instance._client = broken

    results = []
    start = threading.Barrier(5)

    def _worker():
        start.wait(timeout=5.0)
        results.append(instance._run_hindsight_operation(_op_for(broken)))

    threads = [threading.Thread(target=_worker, daemon=True) for _ in range(4)]
    for t in threads:
        t.start()
    start.wait(timeout=5.0)  # release all four retries at once
    for t in threads:
        t.join(timeout=10.0)

    assert sorted(results) == ["ok"] * 4
    assert instance._client is not broken
    # The broken client is replaced exactly once, and the lone replacement survives.
    assert sum(c is not broken for c in built) == 1
    assert [c for c in built if c is not broken and c is not instance._client] == []


def test_shutdown_closes_retired_client_and_allows_rebuild(provider, monkeypatch):
    """shutdown() retires the client under the lock BEFORE closing it, so a
    concurrent ``_get_client()`` rebuilds instead of racing the close, and a later
    ``_get_client()`` must never hand back the closed object."""
    instance, _ = provider({})
    closed = []

    class _CountingClient:
        async def aclose(self):
            closed.append(1)

    retired = _CountingClient()
    instance._client = retired
    rebuilt = types.SimpleNamespace(name="rebuilt")
    monkeypatch.setattr(instance, "_new_cloud_client", lambda: rebuilt)

    instance.shutdown()

    assert closed == [1]
    assert instance._client is None
    assert instance._get_client() is rebuilt
