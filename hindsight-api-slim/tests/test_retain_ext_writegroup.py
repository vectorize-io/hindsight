"""Connection management for the external-backend retain write-group.

The whole point of ``_streaming_batch_write_ext`` is that a store which owns its memory rows
in a separate system must NOT hold the data-plane Postgres connection across the (slow)
object-store write. These tests pin that contract:

* the memory writes (``insert_facts_batch`` + the entity re-posting) happen while NO connection
  is checked out;
* the connection is taken only for the short witness transaction (document/chunk metadata,
  entity-registry reassert, outbox, ``write_txn_witness``);
* ``decide_txn(commit=True)`` publishes the group after the connection is released;
* a later-batch takeover discards the staged write via ``decide_txn(commit=False)`` — and the
  staged write was still connection-free;
* the Postgres link writers (temporal/semantic/causal) are never invoked for an ext org.
"""

from types import SimpleNamespace

import pytest

import hindsight_api.engine.retain.orchestrator as orch
from hindsight_api.engine.retain.types import ConcurrentAppendConflict


class _ConnTracker:
    """Flips ``open`` while a connection is checked out via acquire_with_retry."""

    def __init__(self, current_hash=None):
        self.open = False
        self.store_writes_saw_open = []  # records `open` at each store-write call
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


def _make_common(monkeypatch, tracker, *, calls):
    """Patch the module-level collaborators the helper reaches for."""
    monkeypatch.setattr(orch, "acquire_with_retry", lambda pool: tracker.acquire())

    async def _insert_facts_batch(conn, bank_id, processed, ops=None, txn=None, defer_index=False):
        # defer_index=True mints ids WITHOUT writing — the entity-bearing write happens in
        # index_facts below, so this call is a no-op store-side (records no write).
        calls.append(("insert_facts", conn, tracker.open, defer_index))
        assert conn is None, "ext store write must not receive a connection"
        if not defer_index:
            tracker.store_writes_saw_open.append(tracker.open)
        return ["u1"]

    async def _index_facts(bank_id, unit_ids, facts, document_id=None, unit_entity_ids=None, txn=None):
        # The single, entity-bearing store write — must be connection-free and carry the txn.
        calls.append(("index_facts", tracker.open))
        tracker.store_writes_saw_open.append(tracker.open)
        assert txn is not None, "the deferred store write must ride the write-group txn"

    async def _store_chunks_batch(conn, bank_id, doc_id, meta, ops=None, store_document_text=True):
        calls.append(("store_chunks", tracker.open))
        return {}

    async def _handle_doc_tracking(conn, *a, **k):
        calls.append(("handle_doc_tracking", tracker.open))

    monkeypatch.setattr(orch.fact_storage, "insert_facts_batch", _insert_facts_batch)
    monkeypatch.setattr(orch.fact_storage, "index_facts", _index_facts)
    monkeypatch.setattr(orch.chunk_storage, "store_chunks_batch", _store_chunks_batch)
    monkeypatch.setattr(orch.fact_storage, "handle_document_tracking", _handle_doc_tracking)
    monkeypatch.setattr(orch, "_map_results_to_contents", lambda contents, pf, uids: [list(uids)])
    monkeypatch.setattr(orch, "_remap_phase1_results", lambda rids, e2u, u2e, sem, uids: ([("u1", 0, None)], {}, []))
    # Any PG link writer being called for an ext org is a bug — make it explode.
    for name in ("create_temporal_links_batch", "create_semantic_links_batch", "create_causal_links_batch"):
        if hasattr(orch.link_creation, name):

            def _boom(*a, _n=name, **k):
                raise AssertionError(f"ext path must not call link_creation.{_n}")

            monkeypatch.setattr(orch.link_creation, name, _boom)


class _Provider:
    def __init__(self):
        self.decisions = []
        self.witnesses = []

    def store_owned_for(self, bank_id):
        # False: these tests are about the Protocol-B two-phase write-group — the path for a store
        # whose memory rows live elsewhere but whose document metadata still lands in Postgres. A
        # store that owns its writes skips that dance entirely and has its own tests.
        return False

    async def write_txn_witness(self, txn, *, conn, fq_table):
        self.witnesses.append((txn, conn))

    async def decide_txn(self, txn, *, commit):
        self.decisions.append(commit)


class _EntityResolver:
    def __init__(self, tracker):
        self._t = tracker
        self.postings = []
        self.reasserts = []

    async def record_unit_entity_postings(self, pairs, bank_id=None, store_write=True, txn=None):
        # THE contract: the posting runs with no connection held, whether or not it writes to the
        # store. Both knobs are recorded: `store_write` says whether a store write happens at all,
        # `txn` which write-group it joins when it does.
        assert self._t.open is False, "entity posting must run connection-free"
        self.postings.append((pairs, store_write, txn))

    async def reassert_entities_batch(self, bank_id, resolved, conn):
        assert conn is not None
        self.reasserts.append(bank_id)


def _kwargs(tracker, provider, er, *, doc_tracking_done, existing_hash, new_hash, outbox=None, is_last=False):
    async def _lock(conn, table, doc_id, bank_id):
        return existing_hash

    pool = SimpleNamespace(ops=SimpleNamespace(lock_document_for_write=_lock))
    phase1 = SimpleNamespace(
        entities=SimpleNamespace(
            resolved_entities=[SimpleNamespace(entity_id="e1")],
            entity_to_unit=[(0, 0, None)],
            unit_to_entity_ids={},
        )
    )
    return dict(
        provider=provider,
        ext_txn=SimpleNamespace(txn_id="t1"),
        pool=pool,
        bank_id="bank1",
        fq_table=lambda t: t,
        entity_resolver=er,
        phase1=phase1,
        batch_contents=[{"content": "c"}],
        batch_extracted=[SimpleNamespace(chunk_index=None)],
        batch_processed=[SimpleNamespace(document_id=None, chunk_id=None)],
        batch_chunk_meta=[],
        effective_doc_id="doc1",
        config=SimpleNamespace(store_document_text=True),
        log_buffer=[],
        is_recovery=False,
        is_first_batch=True,
        is_last=is_last,
        doc_tracking_done=doc_tracking_done,
        # Separate latch: `doc_tracking_done` says tracking finished (set even for a zero-unit
        # batch), while this says the document's prior version was actually replaced. These tests
        # drive one batch at a time, so a fresh latch per call is what they want.
        doc_replace_done=[False],
        pipeline_aborted=[False],
        append_base_hash=None,
        new_content_hash=new_hash,
        combined_content="body",
        retain_params=None,
        merged_tags=[],
        outbox_callback=outbox,
        assert_append_base_unchanged=lambda h: None,
        p2_start=0.0,
    )


async def test_store_writes_are_connection_free_and_witness_is_in_txn(monkeypatch):
    tracker = _ConnTracker()
    calls = []
    _make_common(monkeypatch, tracker, calls=calls)
    provider, er = _Provider(), _EntityResolver(tracker)

    kw = _kwargs(tracker, provider, er, doc_tracking_done=[False], existing_hash="__pending__", new_hash="h")
    result = await orch._streaming_batch_write_ext(**kw)

    assert result.aborted is False
    assert result.batch_result_ids == [["u1"]]
    # The fact write happened with no connection held (a single deferred write via index_facts).
    assert tracker.store_writes_saw_open == [False]
    assert ("index_facts", False) in calls  # the entity-bearing store write ran connection-free
    # The posting ran connection-free with `store_write=False`: the entity ids were already
    # written inline by that single `index_facts`, which itself rode `ext_txn`. There is no second
    # store write left to enrol in the group, so no txn is passed — what remains is the Postgres
    # co-occurrence accumulation, which is not a store write at all.
    assert er.postings == [([("u1", "e1", None)], False, None)]
    # Witness written with a real connection, exactly once; commit published after release.
    assert len(provider.witnesses) == 1 and provider.witnesses[0][1] is not None
    assert provider.decisions == [True]
    assert tracker.open is False  # connection released at the end
    # Entity registry reassert ran inside the txn.
    assert er.reasserts == ["bank1"]


async def test_later_batch_takeover_aborts_and_discards_staged_write(monkeypatch):
    tracker = _ConnTracker()
    calls = []
    _make_common(monkeypatch, tracker, calls=calls)
    provider, er = _Provider(), _EntityResolver(tracker)

    # Later batch (doc_tracking already done) whose document was taken over: existing hash
    # differs from ours → abort.
    result = await orch._streaming_batch_write_ext(
        **_kwargs(tracker, provider, er, doc_tracking_done=[True], existing_hash="OTHER", new_hash="OURS")
    )

    assert result.aborted is True
    # Staged store write still happened connection-free before the takeover was detected.
    assert tracker.store_writes_saw_open == [False]
    # The group was explicitly aborted, not committed.
    assert provider.decisions == [False]


async def test_lost_append_race_discards_staged_write(monkeypatch):
    tracker = _ConnTracker()
    calls = []
    _make_common(monkeypatch, tracker, calls=calls)
    provider, er = _Provider(), _EntityResolver(tracker)

    # First batch of an append whose base compare-and-swap fails: the conflict must
    # propagate AND the already-staged store writes must be discarded, exactly once.
    kwargs = _kwargs(tracker, provider, er, doc_tracking_done=[False], existing_hash="MOVED", new_hash="h")
    kwargs["append_base_hash"] = "BASE"

    def _cas_fails(existing_hash):
        raise ConcurrentAppendConflict("append base changed")

    kwargs["assert_append_base_unchanged"] = _cas_fails

    with pytest.raises(ConcurrentAppendConflict):
        await orch._streaming_batch_write_ext(**kwargs)

    assert provider.decisions == [False]
    assert tracker.open is False  # connection released on the way out


async def test_outbox_row_rides_the_connection(monkeypatch):
    tracker = _ConnTracker()
    calls = []
    _make_common(monkeypatch, tracker, calls=calls)
    provider, er = _Provider(), _EntityResolver(tracker)
    seen = {}

    async def _outbox(conn):
        seen["open"] = tracker.open
        seen["conn"] = conn

    result = await orch._streaming_batch_write_ext(
        **_kwargs(
            tracker,
            provider,
            er,
            doc_tracking_done=[False],
            existing_hash="__pending__",
            new_hash="h",
            outbox=_outbox,
            is_last=True,
        )
    )
    assert result.aborted is False
    assert seen["open"] is True and seen["conn"] is not None  # outbox wrote inside the txn
    assert provider.decisions == [True]


# --------------------------------------------------------------------------------------------
# Delta re-retain path
# --------------------------------------------------------------------------------------------


def _make_common_delta(monkeypatch, tracker, *, calls):
    _make_common(monkeypatch, tracker, calls=calls)

    async def _store_document_bodies(*a, **k):
        calls.append(("store_document_bodies", tracker.open))
        assert tracker.open is False, "document-body store write must be connection-free"

    async def _upsert_document_metadata(conn, *a, **k):
        calls.append(("upsert_document_metadata", tracker.open))

    async def _delete_chunks_by_ids(conn, ids, bank_id=None, txn=None, ops=None):
        calls.append(("delete_chunks", tracker.open))
        return 0

    async def _update_meta(conn, *a, **k):
        return 0

    monkeypatch.setattr(orch, "_store_document_bodies", _store_document_bodies)
    monkeypatch.setattr(orch.fact_storage, "upsert_document_metadata", _upsert_document_metadata)
    monkeypatch.setattr(orch.chunk_storage, "delete_chunks_by_ids", _delete_chunks_by_ids)
    monkeypatch.setattr(orch.fact_storage, "update_memory_units_metadata_and_tags", _update_meta)


def _delta_kwargs(tracker, provider, er, *, doc_hash_at_load):
    phase1 = SimpleNamespace(
        entities=SimpleNamespace(
            resolved_entities=[SimpleNamespace(entity_id="e1")],
            entity_to_unit=[(0, 0, None)],
            unit_to_entity_ids={},
        )
    )
    return dict(
        provider=provider,
        ext_txn=SimpleNamespace(txn_id="t1"),
        pool=SimpleNamespace(ops=SimpleNamespace()),
        bank_id="bank1",
        fq_table=lambda t: t,
        entity_resolver=er,
        phase1=phase1,
        effective_doc_id="doc1",
        config=SimpleNamespace(store_document_text=True),
        log_buffer=[],
        processed_facts=[_fact()],
        extracted_facts=[SimpleNamespace(chunk_index=None)],
        delta_contents=[{"content": "c"}],
        contents_dicts=[{"content": "c"}],
        document_tags=[],
        document_body_override=None,
        doc_hash_at_load=doc_hash_at_load,
        new_chunk_metadata=[],
        delta_chunk_map={},
        new_chunks_with_contents={},
        existing_by_index={},
        changed_indices=[],
        removed_indices=[],
        outbox_callback=None,
    )


async def test_a_store_owned_delta_holds_no_connection_and_scopes_its_replace(monkeypatch):
    """The store-owned delta: one `retain`, scoped to the chunks that moved, no connection held.

    This replaces the two Protocol-B delta tests that used to live here. That path is gone — it
    wrote with the plain batch write and tombstoned separately, under a write-group handle nothing
    mints any more — so its witness/decide/re-posting assertions have nothing left to describe. The
    contract that survives is the one this file exists for: the slow writes (the document bodies and
    the retain) must happen with NO Postgres connection checked out, or every concurrent retain
    serialises on the pool.
    """
    tracker = _ConnTracker()
    saw_open = []
    retained = {}

    async def _store_bodies(**kw):
        saw_open.append(tracker.open)
        return None

    class _StoreOwned:
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
    # Nothing slow ran while a connection was checked out.
    assert saw_open == [False, False, False], saw_open
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
