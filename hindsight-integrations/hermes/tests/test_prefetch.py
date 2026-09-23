"""Prefetch slot fencing: the background warm worker's generation fence and
session-owner gate, keeping one session's recall out of another session's slot.

Ported from NousResearch/hermes-agent (PRs #64745 and #117244) to the standalone
plugin; the tests reuse the recording FakeClient from ``conftest`` (no daemon).
"""

import threading

from conftest import FakeClient, FakeRecallResponse


def test_stale_worker_cannot_publish_after_switch_join_timeout(provider, monkeypatch):
    """A prefetch worker can legitimately outlive on_session_switch's 3s join (10s
    drain + 120s recall). Its late publish lands in the NEW session's slot; the
    generation fence must drop it, and the new session's first prefetch() must
    inject nothing of the old one."""
    instance, _ = provider({})
    gate = threading.Event()
    entered = threading.Event()

    def _gated_recall(query):
        entered.set()
        gate.wait(timeout=10.0)
        return "- old-session memory", 1

    monkeypatch.setattr(instance, "_do_recall", _gated_recall)
    instance.queue_prefetch("old session query")
    assert entered.wait(timeout=5.0), "prefetch worker never reached the recall"

    # The worker is still gated when the switch's 3.0s join times out, so the
    # switch returns with the worker alive — the leaked window upstream.
    instance.on_session_switch("new-sid")
    gate.set()
    instance._prefetch_thread.join(timeout=5.0)

    assert instance._prefetch_result == ""
    assert instance.prefetch("anything") == ""
    instance.shutdown()


def test_superseded_worker_cannot_overwrite_newer_result(provider, monkeypatch):
    """Two overlapping workers: the older finishing after the newer must not
    clobber the newer result sitting in the slot."""
    instance, _ = provider({})
    gate_first = threading.Event()
    entered_first = threading.Event()

    def _routed_recall(query):
        if query == "first query":
            entered_first.set()
            gate_first.wait(timeout=10.0)
            return "- first (older) memory", 1
        return "- second (newer) memory", 1

    monkeypatch.setattr(instance, "_do_recall", _routed_recall)

    instance.queue_prefetch("first query")
    worker_first = instance._prefetch_thread
    # Pin the overlap: the older worker must be mid-recall before the newer
    # spawns, so its final publish really exercises the fence (entry barrier
    # added over the upstream port to keep the interleaving deterministic).
    assert entered_first.wait(timeout=5.0), "first worker never reached the recall"
    # Simulate the lost-thread-handle hazard: upstream overwrote
    # _prefetch_thread freely, so a dead handle lets the next warm spawn.
    instance._prefetch_thread = None
    instance.queue_prefetch("second query")
    worker_second = instance._prefetch_thread

    # The newer worker finishes first and wins the slot.
    worker_second.join(timeout=5.0)
    with instance._prefetch_lock:
        assert instance._prefetch_result == "- second (newer) memory"

    # The older worker finishes LAST; its result must be fenced out.
    gate_first.set()
    worker_first.join(timeout=5.0)
    with instance._prefetch_lock:
        assert instance._prefetch_result == "- second (newer) memory"
    instance.shutdown()


def test_shutdown_fences_inflight_prefetch_publish(provider):
    """A recall resuming after shutdown() began must not publish, and the fence
    must prevent a recall against a client shutdown already closed."""
    instance, fake = provider({})
    gate = threading.Event()
    entered = threading.Event()

    async def _gated_arecall(**kwargs):
        fake.recalls.append(kwargs)
        entered.set()
        gate.wait(timeout=10.0)
        return FakeRecallResponse(["late memory"])

    fake.arecall = _gated_arecall

    instance.queue_prefetch("q")
    assert entered.wait(timeout=5.0), "prefetch worker never reached the recall"

    # Release the recall the moment shutdown starts so shutdown's prefetch
    # join returns promptly instead of burning its 5s budget.
    def _release_on_shutdown():
        instance._shutting_down.wait(timeout=10.0)
        gate.set()

    threading.Thread(target=_release_on_shutdown, daemon=True).start()
    instance.shutdown()
    instance._prefetch_thread.join(timeout=5.0)

    assert instance._prefetch_result == ""
    assert instance._client is None
    assert instance.prefetch("q") == ""


def test_queue_prefetch_skips_while_prior_worker_running(provider):
    """Rapid turns must warm serially: while one worker runs, further
    queue_prefetch calls neither spawn nor bump — only ONE recall happens."""
    instance, fake = provider({})
    gate = threading.Event()
    entered = threading.Event()

    async def _gated_arecall(**kwargs):
        fake.recalls.append(kwargs)
        entered.set()
        gate.wait(timeout=10.0)
        return FakeRecallResponse(["m"])

    fake.arecall = _gated_arecall

    instance.queue_prefetch("q1")
    assert entered.wait(timeout=5.0), "prefetch worker never reached the recall"
    first = instance._prefetch_thread

    instance.queue_prefetch("q2")
    instance.queue_prefetch("q3")

    gate.set()
    first.join(timeout=5.0)

    # The live worker was never replaced and nothing else ever recalled.
    assert instance._prefetch_thread is first
    assert len(fake.recalls) == 1
    instance.shutdown()


def test_same_session_boundary_still_publishes(provider):
    """Positive control: with no session switch the fence must be a no-op — an
    uninterrupted warm → prefetch cycle delivers exactly as upstream."""
    instance, _ = provider({}, client=FakeClient(recall_texts=["Memory 1", "Memory 2"]))
    instance.queue_prefetch("test")
    if instance._prefetch_thread:
        instance._prefetch_thread.join(timeout=5.0)

    result = instance.prefetch("test")
    assert "Memory 1" in result
    assert "Memory 2" in result
    instance.shutdown()


def test_late_prefetch_for_superseded_session_is_dropped(provider):
    """Compression switches sessions inline, bypassing the manager's serialized
    boundary task, so a prefetch queued at the previous turn can drain after the
    rotation. Its recall belongs to a session that no longer owns the slot."""
    instance, fake = provider({}, client=FakeClient(recall_texts=["Memory 1", "Memory 2"]))
    # The previous turn's prefetch, still queued on the memory worker.
    instance.queue_prefetch("old session query", session_id="parent-sid")
    if instance._prefetch_thread:
        instance._prefetch_thread.join(timeout=5.0)

    # The inline switch (compression) rotates the slot's owner.
    instance.on_session_switch("child-sid", parent_session_id="parent-sid")

    # The queued task now drains: same query, superseded session.
    instance.queue_prefetch("old session query", session_id="parent-sid")
    if instance._prefetch_thread:
        instance._prefetch_thread.join(timeout=5.0)

    assert instance._prefetch_result == ""
    assert instance.prefetch("child query") == ""
    # And the recall was never spent on a session nobody will read.
    assert fake.recalls == []
    instance.shutdown()


def test_post_switch_prefetch_for_current_session_still_publishes(provider):
    """The session gate must not degenerate into 'drop everything': the new
    session's own prefetch still warms the slot, and callers without a
    session id stay unconstrained."""
    instance, _ = provider({}, client=FakeClient(recall_texts=["Memory 1", "Memory 2"]))
    instance.on_session_switch("child-sid", parent_session_id="parent-sid")

    instance.queue_prefetch("child query", session_id="child-sid")
    if instance._prefetch_thread:
        instance._prefetch_thread.join(timeout=5.0)
    assert "Memory 1" in instance.prefetch("child query")

    instance.queue_prefetch("child query")  # empty session_id = unconstrained
    if instance._prefetch_thread:
        instance._prefetch_thread.join(timeout=5.0)
    assert "Memory 1" in instance.prefetch("child query")
    instance.shutdown()


def test_sync_turn_revert_cannot_blind_the_session_gate(provider):
    """sync_turn re-stamps _session_id from its kwarg; a queued sync for the
    previous session lands after an inline switch and un-rotates it. The gate
    must key off a boundary-owned id, not _session_id."""
    instance, _ = provider({}, client=FakeClient(recall_texts=["Memory 1", "Memory 2"]))
    instance.queue_prefetch("old session query", session_id="parent-sid")
    if instance._prefetch_thread:
        instance._prefetch_thread.join(timeout=5.0)

    instance.on_session_switch("child-sid", parent_session_id="parent-sid")
    # Queued sync_all(parent) drains late and reverts _session_id.
    instance.sync_turn("x", "y", session_id="parent-sid")
    assert instance._session_id == "parent-sid"  # the un-rotation is real

    instance.queue_prefetch("old session query", session_id="parent-sid")
    if instance._prefetch_thread:
        instance._prefetch_thread.join(timeout=5.0)

    assert instance._prefetch_result == ""
    assert instance.prefetch("child query") == ""
    instance.shutdown()
