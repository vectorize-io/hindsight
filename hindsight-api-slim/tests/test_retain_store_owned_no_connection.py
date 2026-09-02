"""Connection management for a store-owned retain write.

A store that owns its memory rows keeps them in a separate system, and the write to it is slow.
It must NOT happen with the data-plane Postgres connection checked out, or every concurrent
retain serialises on the pool. Both store-owned write paths — the streaming batch write and the
delta re-retain — are pinned here:

* the document bodies, the id mint and the ``retain`` RPC all run with NO connection held;
* a delta names the chunks that moved rather than replacing the whole document;
* a delta that loses its watermark compare-and-set writes nothing and falls back.
"""

from types import SimpleNamespace

import hindsight_api.engine.retain.orchestrator as orch
from hindsight_api.engine.retain.types import ConcurrentAppendConflict


class _ConnTracker:
    """Flips ``open`` while a connection is checked out via acquire_with_retry."""

    def __init__(self, current_hash=None):
        self.open = False
        self._current_hash = current_hash

    def acquire(self):
        tracker = self

        async def _fetchval(*a, **k):
            return tracker._current_hash

        class _CM:
            async def __aenter__(self_inner):
                tracker.open = True
                return SimpleNamespace(name="conn", transaction=_txn, fetchval=_fetchval)

            async def __aexit__(self_inner, *a):
                tracker.open = False
                return False

        return _CM()


def _txn():
    class _T:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *a):
            return False

    return _T()


async def _noop_async(*a, **k):
    return None


def _fact():
    """The fields the entity-name merge reaches for on a processed fact."""
    return SimpleNamespace(
        document_id=None,
        chunk_id=None,
        content_index=0,
        fact_text="hello",
        entities=[],
        occurred_start=None,
        occurred_end=None,
        mentioned_at=None,
    )


# --------------------------------------------------------------------------------------------
# Streaming batch write
# --------------------------------------------------------------------------------------------


async def test_a_store_owned_batch_write_holds_no_connection(monkeypatch):
    """The whole batch — mint, then one server-side `retain` — runs connection-free.

    There is no connection phase at all on this path: no ``documents``/``chunks``/``entities``
    rows are written, so nothing needs the pool. The test acquires nothing and asserts the
    tracker never opened, which is stronger than "released it in time".
    """
    tracker = _ConnTracker()
    saw_open = []
    retained = {}

    class _StoreOwned:
        async def retain(self, bank_id, unit_ids, facts, **kw):
            saw_open.append(tracker.open)
            retained.update(kw)
            return SimpleNamespace(seq=3, new_entities=1)

    async def _insert(conn, *a, **k):
        assert conn is None, "the store write must not receive a connection"
        saw_open.append(tracker.open)
        return ["u1"]

    monkeypatch.setattr(orch.fact_storage, "insert_facts_batch", _insert)
    monkeypatch.setattr(orch, "acquire_with_retry", lambda *_a, **_k: tracker.acquire())

    doc_tracking_done = [False]
    result_ids = await orch._streaming_store_owned_retain(
        provider=_StoreOwned(),
        pool=SimpleNamespace(ops=None),
        bank_id="b",
        batch_contents=[SimpleNamespace(entities=None, resolve_entities=True)],
        batch_extracted=[SimpleNamespace(chunk_index=None)],
        batch_processed=[_fact()],
        batch_chunk_meta=[],
        effective_doc_id="d1",
        config=SimpleNamespace(entity_similarity_threshold=0.0),
        log_buffer=[],
        is_first_batch=True,
        append_base_hash=None,
        doc_tracking_done=doc_tracking_done,
        doc_replace_done=[False],
        p2_start=0.0,
    )

    assert result_ids == [["u1"]]
    assert saw_open == [False, False], saw_open
    # The first batch replaces the document's prior version, in the same atomic entry.
    assert retained["replace_document_id"] == "d1"
    # Tracking is complete here: the post-loop finalizer must not write a Postgres documents row
    # and tombstone by document_id at a later sequence, which would delete what we just wrote.
    assert doc_tracking_done == [True]


# --------------------------------------------------------------------------------------------
# Delta re-retain path
# --------------------------------------------------------------------------------------------


async def test_a_store_owned_delta_holds_no_connection_and_scopes_its_replace(monkeypatch):
    """The store-owned delta: one `retain`, scoped to the chunks that moved, no connection held."""
    tracker = _ConnTracker()
    saw_open = []
    retained = {}

    async def _store_bodies(**kw):
        saw_open.append(tracker.open)
        return None

    class _StoreOwned:
        def store_owned_for(self, bank_id):
            return True

        async def scan_memories(self, **kw):
            # The observation sweep asks the store which facts the replaced chunks own. Answering
            # "none" is the interesting case here: it must still go through the STORE, never the
            # SQL read, which is what `saw_open` staying three entries proves.
            saw_open.append(tracker.open)
            return SimpleNamespace(memories=[], next_page_token="")

        async def retain(self, bank_id, unit_ids, facts, **kw):
            saw_open.append(tracker.open)
            retained.update(kw)
            retained["unit_ids"] = list(unit_ids)
            return SimpleNamespace(seq=7, new_entities=0)

    async def _insert(*a, **k):
        saw_open.append(tracker.open)
        return ["u1"]

    monkeypatch.setattr(orch, "_store_document_bodies", _store_bodies)
    monkeypatch.setattr(orch.fact_storage, "insert_facts_batch", _insert)
    monkeypatch.setattr(orch, "acquire_with_retry", lambda *_a, **_k: tracker.acquire())

    ok, unit_ids = await orch._delta_store_owned_write(
        provider=_StoreOwned(),
        pool=SimpleNamespace(ops=None),
        bank_id="b",
        effective_doc_id="d1",
        config=SimpleNamespace(entity_similarity_threshold=0.0),
        log_buffer=[],
        entity_resolver=SimpleNamespace(flush_pending_stats=_noop_async),
        contents_dicts=[{"content": "hello"}],
        delta_contents=[SimpleNamespace(entities=None, resolve_entities=True)],
        document_tags=[],
        document_body_override=None,
        extracted_facts=[SimpleNamespace(chunk_index=0)],
        processed_facts=[_fact()],
        new_chunk_metadata=[SimpleNamespace(chunk_index=0)],
        delta_chunk_map={},
        new_chunks_with_contents={0: "hello"},
        existing_by_index={1: SimpleNamespace(chunk_id="b_d1_1")},
        changed_indices=[1],
        removed_indices=[],
        doc_watermark_at_load=5,
    )

    assert ok is True
    assert unit_ids == [["u1"]]
    # Nothing slow ran while a connection was checked out — the observation sweep's scan included.
    assert saw_open == [False, False, False, False], saw_open
    # And the replace was SCOPED — the changed chunk named, not the whole document blown away.
    assert retained["replace_document_id"] == "d1"
    assert retained["replace_chunk_ids"] == ["b_d1_1"]


async def test_a_store_owned_delta_falls_back_when_the_document_moved(monkeypatch):
    """The watermark compare-and-set is the fence, and losing it means falling back.

    `_store_document_bodies` runs FIRST precisely so this is detected before anything is written:
    it compare-and-sets on the document's watermark, and the fact write would move the WAL head
    that CAS reads. Fencing after the write would fence the batch against itself.
    """
    calls = []

    async def _store_bodies(**kw):
        calls.append("store_bodies")
        raise ConcurrentAppendConflict("moved")

    async def _insert(*a, **k):
        calls.append("insert")
        return ["u1"]

    class _StoreOwned:
        async def retain(self, *a, **k):
            calls.append("retain")
            return SimpleNamespace(seq=1, new_entities=0)

    monkeypatch.setattr(orch, "_store_document_bodies", _store_bodies)
    monkeypatch.setattr(orch.fact_storage, "insert_facts_batch", _insert)

    ok, unit_ids = await orch._delta_store_owned_write(
        provider=_StoreOwned(),
        pool=SimpleNamespace(ops=None),
        bank_id="b",
        effective_doc_id="d1",
        config=SimpleNamespace(entity_similarity_threshold=0.0),
        log_buffer=[],
        entity_resolver=SimpleNamespace(flush_pending_stats=_noop_async),
        contents_dicts=[{"content": "hello"}],
        delta_contents=[SimpleNamespace(entities=None, resolve_entities=True)],
        document_tags=[],
        document_body_override=None,
        extracted_facts=[SimpleNamespace(chunk_index=0)],
        processed_facts=[_fact()],
        new_chunk_metadata=[SimpleNamespace(chunk_index=0)],
        delta_chunk_map={},
        new_chunks_with_contents={0: "hello"},
        existing_by_index={1: SimpleNamespace(chunk_id="b_d1_1")},
        changed_indices=[1],
        removed_indices=[],
        doc_watermark_at_load=5,
    )

    assert ok is False
    assert unit_ids == []
    # Nothing was written: the fence tripped before the fact write, which is the point of it
    # running first.
    assert calls == ["store_bodies"], calls
