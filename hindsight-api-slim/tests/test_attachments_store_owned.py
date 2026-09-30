"""A store-owned bank's per-fact attachments come off the rows the store returned.

Recall, list-memories and get-memory report the attachments each fact was drawn from. For a
Postgres-backed bank the ids are read from ``memory_units.attachment_ids``. For a bank whose
memories store owns its rows that table holds none of them, and reading it is not a cheap empty
read: the table carries partial vector indexes per bank, and the planner opens and locks every
index on a table to plan any statement against it -- in a tenant with a few thousand banks, ~15k
locks and hundreds of milliseconds of planning, on every recall.

So the store carries the ids on what it already returns (recall rows, list/detail items), and the
read surfaces resolve them from there. The properties pinned here:

* the ids survive the write model (``FactRecord.metadata_bag``) and the retain pipeline's record
  builder;
* the lookup for a store-owned bank never plans a statement against ``memory_units``, and with
  nothing carried it returns before asking for the bank profile or a connection at all;
* every read surface actually hands back the ``attachments`` for such a bank -- asserted on the
  HTTP payload, over a bank whose ``memory_units`` holds nothing, so the only way to produce them
  is from the carried ids.
"""

import base64
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import hindsight_api.engine.memories as memories_module
from hindsight_api.engine.chunk_ids import build_chunk_id
from hindsight_api.engine.memories import set_memories
from hindsight_api.engine.memories.base import (
    DOC_META_ATTACHMENT_FILENAMES,
    META_ATTACHMENT_IDS,
    FactRecord,
    StoredMemory,
    build_fact_records,
    document_attachment_filenames,
    document_record_metadata,
)
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.response_models import MemoryFact, RecallResult
from hindsight_api.engine.retain.attachment_content import (
    attachment_placeholder,
    compute_attachment_hash,
    short_attachment_id,
)
from hindsight_api.engine.retain.attachment_store import StoredAttachment, _record_attachments
from tests.test_memories_extension import InMemoryMemories

UNIT_A = "00000000-0000-0000-0000-00000000000a"
UNIT_B = "00000000-0000-0000-0000-00000000000b"
UNIT_PLAIN = "00000000-0000-0000-0000-0000000000cc"
# A real hash/short-id pair, so a placeholder written into document or chunk text resolves.
SHOT_HASH = compute_attachment_hash(b"the vpn reset screenshot")
SHOT = short_attachment_id(SHOT_HASH)
DIAGRAM = "0f1e2d3c4b5a"
DOCUMENT_ID = "vpn-article"
SHOT_NAME = "vpn-reset.png"


# -- the write model ---------------------------------------------------------


def test_metadata_bag_carries_attachment_ids_deduplicated_in_order():
    record = FactRecord(
        unit_id=UNIT_A, text="t", embedding=[0.0], fact_type="world", attachment_ids=[SHOT, DIAGRAM, SHOT]
    )

    bag = record.metadata_bag()

    assert json.loads(bag[META_ATTACHMENT_IDS]) == [SHOT, DIAGRAM]


def test_metadata_bag_omits_the_key_for_a_fact_with_no_attachments():
    """A plain-text fact is the overwhelmingly common case: no key, not an empty list."""
    record = FactRecord(unit_id=UNIT_A, text="t", embedding=[0.0], fact_type="world")

    assert META_ATTACHMENT_IDS not in record.metadata_bag()


def test_build_fact_records_carries_each_facts_attachment_ids():
    """The builder the store-owned retain paths use must not drop the extractor's attribution."""

    def _fact(ids):
        return SimpleNamespace(
            fact_text="t",
            embedding=[0.0],
            fact_type="world",
            tags=[],
            context=None,
            document_id="doc",
            chunk_id=None,
            metadata=None,
            observation_scopes=None,
            entities=[],
            causal_relations=[],
            occurred_start=None,
            occurred_end=None,
            mentioned_at=None,
            attachment_ids=ids,
        )

    records = build_fact_records([UNIT_A, UNIT_PLAIN], [_fact([SHOT]), _fact([])])

    assert [r.attachment_ids for r in records] == [[SHOT], []]
    assert json.loads(records[0].metadata_bag()[META_ATTACHMENT_IDS]) == [SHOT]


# -- the document record's names ---------------------------------------------


def test_document_record_metadata_round_trips_the_names():
    """The names are a JSON string in a string -> string map, next to the retain params."""
    metadata = document_record_metadata({"context": "kb"}, {SHOT: SHOT_NAME, DIAGRAM: ""})

    assert all(isinstance(v, str) for v in metadata.values())
    assert json.loads(metadata["retain_params"]) == {"context": "kb"}
    # An empty name is no name: it must not come back as a filename of "".
    assert document_attachment_filenames({"metadata": metadata}) == {SHOT: SHOT_NAME}


def test_document_record_metadata_omits_the_key_when_nothing_is_named():
    assert document_record_metadata(None, {}) == {}
    assert DOC_META_ATTACHMENT_FILENAMES not in document_record_metadata({"context": "kb"}, None)


@pytest.mark.parametrize(
    "record",
    [None, {}, {"metadata": {}}, {"metadata": {DOC_META_ATTACHMENT_FILENAMES: "{not json"}}],
    ids=["absent", "no-metadata", "no-key", "unparseable"],
)
def test_a_record_without_readable_names_has_none(record):
    """A record written before the key existed reads back exactly as it did: no names."""
    assert document_attachment_filenames(record) == {}


@pytest.mark.asyncio
async def test_the_document_body_write_carries_the_names(monkeypatch):
    """`_store_document_bodies` is the writer the delta, non-session and batched paths share."""
    from hindsight_api.engine.retain import orchestrator

    written: list[dict] = []

    class _Store:
        def store_owned_for(self, bank_id):
            return True

        async def put_document(self, **kwargs):
            written.append(kwargs)

    monkeypatch.setattr(memories_module, "get_memories", lambda: _Store())

    await orchestrator._store_document_bodies(
        bank_id="bank-1",
        document_id=DOCUMENT_ID,
        combined_content="body",
        chunk_texts=["body"],
        merged_tags=[],
        config=SimpleNamespace(store_document_text=True),
        content_hash="h",
        retain_params={"context": "kb"},
        attachment_filenames={SHOT: SHOT_NAME},
    )

    assert document_attachment_filenames(written[0]) == {SHOT: SHOT_NAME}
    assert json.loads(written[0]["metadata"]["retain_params"]) == {"context": "kb"}


# -- the engine lookup -------------------------------------------------------


class _NoPostgres:
    """Stands in for the engine: any attempt to reach Postgres fails the test."""

    async def get_bank_profile(self, *a, **k):
        raise AssertionError("a store-owned bank's attachment lookup read the bank profile")

    async def _get_backend(self):
        raise AssertionError("a store-owned bank's attachment lookup took a connection")


class _AttachmentsOnlyConn:
    """Answers the attachment-table read; fails on any statement that names ``memory_units``."""

    def __init__(self):
        self.statements: list[str] = []

    async def fetch(self, sql, bank_id, ids, document_id=None):
        self.statements.append(sql)
        assert "memory_units" not in sql, "a store-owned bank's attachment lookup planned against memory_units"
        known = {
            SHOT: ("h" * 64, "image/png", 10, "image"),
            DIAGRAM: ("d" * 64, "image/svg+xml", 20, "image"),
        }
        # The name is on the row, per document: doc-1 named the screenshot, doc-2 did not.
        return [
            {
                "attachment_hash": known[i][0],
                "short_id": i,
                "media_type": known[i][1],
                "byte_size": known[i][2],
                "storage_key": f"k/{i}",
                "kind": known[i][3],
                "filename": SHOT_NAME if (i == SHOT and document_id == "doc-1") else None,
            }
            for i in ids
            if i in known
        ]


class _EngineWithConn:
    def __init__(self, conn):
        self.conn = conn

    async def get_bank_profile(self, *a, **k):
        return {"bank_id": "bank-1"}

    async def _get_backend(self):
        conn = self.conn

        class _Backend:
            @asynccontextmanager
            async def acquire(self):
                yield conn

        return _Backend()


def _memories(store_owned: bool, names: "dict[str, dict[str, str]] | None" = None):
    """A store stub. ``names`` is document_id -> attachment names on that document's record;
    ``record_reads`` logs every batched record read, so a test can count them."""
    record_reads: list[list[str]] = []

    async def get_document_records(*, bank_id, document_ids):
        record_reads.append(list(document_ids))
        return {
            d: {"document_id": d, "metadata": document_record_metadata(None, (names or {})[d])}
            for d in document_ids
            if d in (names or {})
        }

    return SimpleNamespace(
        store_owned_for=lambda bank_id: store_owned,
        get_document_records=get_document_records,
        record_reads=record_reads,
    )


# -- reclaim -----------------------------------------------------------------


class _ReclaimEngine:
    """Enough of the engine for the reclaim path; its file storage fails on any delete."""

    _unreferenced_attachment_blobs = MemoryEngine._unreferenced_attachment_blobs

    class _Files:
        async def delete(self, key):
            raise AssertionError(f"reclaim deleted the blob {key}, which a row still names")

    _file_storage = _Files()


class _ReclaimConn:
    """Answers the two reads the reclaim makes, and records the delete."""

    def __init__(self, keys: list[str], still_named: "list[str] | None" = None):
        self._keys = keys
        self._still_named = still_named or []
        self.deleted: list[tuple] = []

    async def fetch(self, sql, *args):
        if "DISTINCT storage_key" in sql:
            return [{"storage_key": key} for key in self._still_named]
        return [{"storage_key": key} for key in self._keys]

    async def execute(self, sql, *args):
        self.deleted.append(args)


@pytest.mark.asyncio
async def test_reclaim_runs_for_a_store_owned_bank():
    """#4462: an attachment belongs to its document, so this is the same on both backends.

    It used to bail out here. A store-owned bank's documents are not SQL rows, so it had no
    reference rows, and "no reference survives" could not tell an unreferenced attachment from
    one a store-held document still shows -- which meant nothing ever reclaimed these banks'
    attachments. Now the attachment row names its own document, and there is nothing to ask.
    """
    conn = _ReclaimConn([f"k/{SHOT}"])

    orphans = await MemoryEngine._drop_orphaned_attachments(_ReclaimEngine(), conn, "bank-1", DOCUMENT_ID)

    assert orphans == [f"k/{SHOT}"]
    assert conn.deleted == [("bank-1", DOCUMENT_ID)], "the document's own rows, and only those"


@pytest.mark.asyncio
async def test_reclaim_keeps_a_blob_another_row_still_names():
    """A bank retained before this rule shares one blob across its documents.

    The migration gives each document a row of its own but does not move the bytes, so the
    surviving rows still name that key; freeing it with the first document would blank the
    image in the others.
    """
    shared = f"legacy/{SHOT}"
    conn = _ReclaimConn([shared], still_named=[shared])

    assert await MemoryEngine._drop_orphaned_attachments(_ReclaimEngine(), conn, "bank-1", DOCUMENT_ID) == []


@pytest.mark.asyncio
async def test_a_store_owned_bank_resolves_no_attachments_without_touching_postgres(monkeypatch):
    """With nothing carried there is nothing to resolve, so not even the bank profile is read.

    The property is that the lookup returns before it asks for the bank profile or a connection,
    not merely that it returns ``{}``: an empty result is what the expensive read produced too.
    """
    store = _memories(store_owned=True, names={"doc-1": {SHOT: SHOT_NAME}})
    monkeypatch.setattr(memories_module, "get_memories", lambda: store)

    result = await MemoryEngine.attachments_for_memories(_NoPostgres(), "bank-1", [UNIT_A], request_context=None)

    assert result == {}
    # Nor the store: a response with no attachments pays nothing for their names.
    assert store.record_reads == []


@pytest.mark.asyncio
async def test_a_store_owned_bank_resolves_the_carried_ids_without_memory_units(monkeypatch):
    store = _memories(store_owned=True, names={"doc-1": {SHOT: SHOT_NAME}})
    monkeypatch.setattr(memories_module, "get_memories", lambda: store)
    conn = _AttachmentsOnlyConn()

    result = await MemoryEngine.attachments_for_memories(
        _EngineWithConn(conn),
        "bank-1",
        [UNIT_A, UNIT_B, UNIT_PLAIN],
        request_context=None,
        carried={
            UNIT_A: ("doc-1", [SHOT, DIAGRAM]),
            UNIT_B: ("doc-2", [DIAGRAM]),
            UNIT_PLAIN: ("doc-1", []),
            # Carried for a unit this page is not rendering: not resolved, not returned.
            "00000000-0000-0000-0000-0000000000ff": ("doc-1", [SHOT]),
        },
    )

    assert {unit: [r.short_id for r in records] for unit, records in result.items()} == {
        UNIT_A: [SHOT, DIAGRAM],
        UNIT_B: [DIAGRAM],
    }
    assert result[UNIT_A][0].media_type == "image/png"
    # One read per document: the filename is on that document's own attachment row.
    assert len(conn.statements) == 2
    # So the same bytes named in doc-1 stay unnamed in doc-2...
    assert {unit: [r.filename for r in records] for unit, records in result.items()} == {
        UNIT_A: [SHOT_NAME, None],
        UNIT_B: [None],
    }
    # ...and the store's document records are not read at all for them.
    assert store.record_reads == []


class _ProbedConn(_AttachmentsOnlyConn):
    """Also answers the "does this bank hold any attachment" probe an observation lookup makes."""

    def __init__(self, bank_has_attachments: bool = True):
        super().__init__()
        self.bank_has_attachments = bank_has_attachments

    async def fetchval(self, sql, bank_id):
        self.statements.append(sql)
        assert "memory_units" not in sql
        return 1 if self.bank_has_attachments else None


def _observation_store(memories: "list[StoredMemory]"):
    """A store-owned stub answering get-by-id from ``memories``; ``reads`` logs each call's ids."""
    by_id = {m.unit_id: m for m in memories}
    reads: list[list[str]] = []

    async def get_memories(*, conn, fq_table, bank_id, unit_ids):
        reads.append(list(unit_ids))
        return [by_id[u] for u in unit_ids if u in by_id]

    return SimpleNamespace(store_owned_for=lambda bank_id: True, get_memories=get_memories, reads=reads)


OBSERVATION = "00000000-0000-0000-0000-0000000000ob"


@pytest.mark.asyncio
async def test_a_store_owned_observation_shows_its_sources_attachments_from_the_store(monkeypatch):
    """Nothing is carried for an observation, so its sources are read through the store, never SQL."""
    store = _observation_store(
        [
            StoredMemory(
                unit_id=OBSERVATION, text="o", fact_type="observation", source_memory_ids=[UNIT_A, UNIT_B, UNIT_PLAIN]
            ),
            StoredMemory(unit_id=UNIT_A, text="a", fact_type="world", document_id="doc-1", attachment_ids=[SHOT]),
            # Same screenshot again, from another fact: shown once.
            StoredMemory(
                unit_id=UNIT_B, text="b", fact_type="world", document_id="doc-2", attachment_ids=[DIAGRAM, SHOT]
            ),
            StoredMemory(unit_id=UNIT_PLAIN, text="p", fact_type="world", document_id="doc-1"),
        ]
    )
    monkeypatch.setattr(memories_module, "get_memories", lambda: store)
    conn = _ProbedConn()

    result = await MemoryEngine.attachments_for_memories(
        _EngineWithConn(conn), "bank-1", [OBSERVATION], request_context=None, observation_ids=[OBSERVATION]
    )

    assert [r.short_id for r in result[OBSERVATION]] == [SHOT, DIAGRAM]
    assert store.reads == [[OBSERVATION], [UNIT_A, UNIT_B, UNIT_PLAIN]]


@pytest.mark.asyncio
async def test_an_observation_in_a_bank_without_attachments_reads_nothing_more(monkeypatch):
    """The common case pays one primary-key probe, and no store read, for its observations."""
    store = _observation_store([])
    monkeypatch.setattr(memories_module, "get_memories", lambda: store)
    conn = _ProbedConn(bank_has_attachments=False)

    result = await MemoryEngine.attachments_for_memories(
        _EngineWithConn(conn), "bank-1", [OBSERVATION], request_context=None, observation_ids=[OBSERVATION]
    )

    assert result == {}
    assert store.reads == []
    assert len(conn.statements) == 1


@pytest.mark.asyncio
async def test_a_bank_whose_rows_live_in_sql_still_reads_them(monkeypatch):
    """The guard must not swallow the Postgres-backed case: there the read is the feature."""
    monkeypatch.setattr(memories_module, "get_memories", lambda: _memories(store_owned=False))

    with pytest.raises(AssertionError, match="read the bank profile"):
        await MemoryEngine.attachments_for_memories(_NoPostgres(), "bank-1", [UNIT_A], request_context=None)


# -- every read surface, end to end -----------------------------------------


class _CarryingStore(InMemoryMemories):
    """A store-owned store that returns each memory's attachment ids on its rows.

    ``answers_full_recall`` picks which recall path is exercised: the store answering the whole
    recall (its rows become ``MemoryFact`` directly), or declining it so the engine fuses the arm
    results (its rows become ``RetrievalResult``). Both must surface the same attachments.
    """

    def __init__(self, answers_full_recall: bool):
        super().__init__({})
        self.answers_full_recall = answers_full_recall
        #: Every batched record read, so a test can count what a response cost.
        self.record_reads: list[list[str]] = []

    async def get_document_records(self, *, bank_id, document_ids):
        self.record_reads.append(list(document_ids))
        return await super().get_document_records(bank_id=bank_id, document_ids=document_ids)

    async def full_recall(self, request):
        if not self.answers_full_recall:
            return None
        return RecallResult(
            results=[
                MemoryFact(
                    id=row.unit_id,
                    text=row.text,
                    fact_type=row.fact_type,
                    document_id=row.document_id,
                    attachment_ids=list(row.attachment_ids),
                )
                for row in self.rows.values()
            ]
        )

    async def search(self, *, conn, bank_id, fact_types, query_embedding, query_text, limit, **kwargs):
        out = await super().search(
            conn=conn,
            bank_id=bank_id,
            fact_types=fact_types,
            query_embedding=query_embedding,
            query_text=query_text,
            limit=limit,
            **kwargs,
        )
        for arms in out.values():
            for result in [*arms.semantic, *arms.bm25]:
                result.attachment_ids = list(self.rows[result.id].attachment_ids)
        return out

    async def get_document_record(self, *, bank_id, document_id, include_text=False):
        # A real store's record carries its write stamps (epoch ms), which the document route
        # requires; the base stub leaves them out because none of its own tests render one.
        record = await super().get_document_record(bank_id=bank_id, document_id=document_id, include_text=include_text)
        if record is not None:
            stamp = int(datetime.now(timezone.utc).timestamp() * 1000)
            record.update(created_at=stamp, updated_at=stamp)
        return record

    def _render(self, row: StoredMemory) -> dict:
        return {
            "id": row.unit_id,
            "text": row.text,
            "fact_type": row.fact_type,
            "document_id": row.document_id,
            "attachment_ids": list(row.attachment_ids),
        }

    async def list_memory_units(self, *, conn, ops, fq_table, bank_id, limit=100, offset=0, **kwargs):
        items = [self._render(row) for row in self.rows.values()]
        return {"items": items[offset : offset + limit], "total": len(items), "limit": limit, "offset": offset}

    async def get_memory_unit(self, *, conn, ops, fq_table, bank_id, unit_id):
        row = self.rows.get(str(unit_id))
        return None if row is None else self._render(row)


@pytest.fixture
def restore_default_store():
    yield
    set_memories(None)


async def _store_owned_bank(
    memory, request_context, answers_full_recall: bool, *, with_attachment: bool = True
) -> tuple[str, _CarryingStore]:
    bank_id = f"so-attach-{uuid.uuid4().hex[:8]}"
    store = _CarryingStore(answers_full_recall)
    set_memories(store)
    # The document record, carrying the name its retain gave the screenshot -- where a store-owned
    # bank keeps what `document_attachments.filename` holds for a SQL one.
    await _seed_document(store, bank_id, keep_text=True)
    # The store owns the facts, never the bank row, so create it the way a retain would.
    await memory.ensure_bank_profile(bank_id, request_context=request_context)
    backend = await memory._get_backend()
    async with backend.acquire() as conn:
        await _record_attachments(
            conn,
            bank_id,
            DOCUMENT_ID,
            [
                StoredAttachment(
                    attachment_hash=SHOT_HASH,
                    short_id=SHOT,
                    media_type="image/png",
                    byte_size=68,
                    storage_key=f"attachments/{bank_id}/{SHOT}",
                    kind="image",
                    filename=SHOT_NAME,
                )
            ],
        )
    now = datetime.now(timezone.utc)
    store.rows[UNIT_A] = StoredMemory(
        unit_id=UNIT_A,
        text="To reset the VPN, click the reset button.",
        fact_type="world",
        document_id="vpn-article",
        created_at=now,
        attachment_ids=[SHOT] if with_attachment else [],
    )
    store.rows[UNIT_PLAIN] = StoredMemory(
        unit_id=UNIT_PLAIN,
        text="The VPN client is called Sentinel.",
        fact_type="world",
        document_id="vpn-article",
        created_at=now,
    )
    return bank_id, store


def _assert_shot(attachments, bank_id: str) -> None:
    assert attachments, "no attachments returned"
    assert [a["id"] for a in attachments] == [SHOT]
    assert attachments[0]["media_type"] == "image/png"
    assert attachments[0]["url"] == f"/v1/default/banks/{bank_id}/attachments/{SHOT}"
    # The name the retain gave it, read back off the document's own attachment row.
    assert attachments[0]["filename"] == SHOT_NAME


@pytest.mark.asyncio
@pytest.mark.parametrize("answers_full_recall", [True, False], ids=["store-answered", "engine-fused"])
async def test_recall_returns_the_attachments_the_store_carried(
    api_client, memory, request_context, restore_default_store, answers_full_recall
):
    bank_id, store = await _store_owned_bank(memory, request_context, answers_full_recall)

    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories/recall",
        json={"query": "how do I reset the VPN", "types": ["world"], "limit": 10},
    )

    assert response.status_code == 200, response.text
    by_id = {r["id"]: r for r in response.json()["results"]}
    assert set(by_id) == {UNIT_A, UNIT_PLAIN}
    _assert_shot(by_id[UNIT_A].get("attachments"), bank_id)
    assert by_id[UNIT_PLAIN].get("attachments") is None
    # The carrier stays internal: the payload is the same shape on either backend.
    assert all("attachment_ids" not in r for r in by_id.values())
    # The names ride on the attachment rows the lookup already reads, so the store's document
    # records cost nothing at all.
    assert store.record_reads == []


@pytest.mark.asyncio
@pytest.mark.parametrize("answers_full_recall", [True, False], ids=["store-answered", "engine-fused"])
async def test_a_response_with_no_attachments_reads_no_document_record(
    api_client, memory, request_context, restore_default_store, answers_full_recall
):
    """The common case -- nothing on the page was drawn from an attachment -- costs nothing extra."""
    bank_id, store = await _store_owned_bank(memory, request_context, answers_full_recall, with_attachment=False)

    recall = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories/recall",
        json={"query": "how do I reset the VPN", "types": ["world"], "limit": 10},
    )
    listed = await api_client.get(f"/v1/default/banks/{bank_id}/memories/list")
    detail = await api_client.get(f"/v1/default/banks/{bank_id}/memories/{UNIT_A}")

    for response in (recall, listed, detail):
        assert response.status_code == 200, response.text
    assert all(not r.get("attachments") for r in recall.json()["results"])
    assert all(not m.get("attachments") for m in listed.json()["items"])
    assert not detail.json().get("attachments")
    assert store.record_reads == []


@pytest.mark.asyncio
async def test_list_and_get_return_the_attachments_the_store_carried(
    api_client, memory, request_context, restore_default_store
):
    bank_id, store = await _store_owned_bank(memory, request_context, answers_full_recall=True)

    listed = await api_client.get(f"/v1/default/banks/{bank_id}/memories/list")
    detail = await api_client.get(f"/v1/default/banks/{bank_id}/memories/{UNIT_A}")
    # The names are on the attachment rows; no document record is read for them.
    assert store.record_reads == []

    assert listed.status_code == 200, listed.text
    items = {m["id"]: m for m in listed.json()["items"]}
    _assert_shot(items[UNIT_A].get("attachments"), bank_id)
    assert items[UNIT_PLAIN].get("attachments") is None
    assert all("attachment_ids" not in m for m in items.values())

    assert detail.status_code == 200, detail.text
    _assert_shot(detail.json().get("attachments"), bank_id)
    assert "attachment_ids" not in detail.json()


# -- documents and chunks ----------------------------------------------------


class _NoChunkTableConn(_AttachmentsOnlyConn):
    """Also fails on any read of the SQL ``chunks`` table, which a store-owned bank never fills."""

    async def fetch(self, sql, bank_id, ids, document_id=None):
        assert " chunks" not in sql and '"chunks"' not in sql and ".chunks" not in sql, (
            "a store-owned bank's chunk attachment lookup read the SQL chunks table"
        )
        return await super().fetch(sql, bank_id, ids, document_id)


@pytest.mark.asyncio
async def test_a_store_owned_chunk_with_no_placeholder_touches_no_postgres(monkeypatch):
    monkeypatch.setattr(memories_module, "get_memories", lambda: _memories(store_owned=True))

    result = await MemoryEngine.attachments_for_chunks(
        _NoPostgres(), "bank-1", ["c0"], request_context=None, carried_texts={"c0": ("doc-1", "plain prose")}
    )

    assert result == {}


@pytest.mark.asyncio
async def test_a_store_owned_chunk_resolves_from_the_carried_text(monkeypatch):
    monkeypatch.setattr(memories_module, "get_memories", lambda: _memories(store_owned=True))
    conn = _NoChunkTableConn()

    result = await MemoryEngine.attachments_for_chunks(
        _EngineWithConn(conn),
        "bank-1",
        ["c0", "c1"],
        request_context=None,
        carried_texts={
            "c0": ("doc-1", f"click {attachment_placeholder(SHOT_HASH)} then reconnect"),
            "c1": ("doc-1", "no image here"),
        },
    )

    assert {chunk: [r.short_id for r in records] for chunk, records in result.items()} == {"c0": [SHOT]}
    # The name comes off doc-1's own attachment row, not from the store.
    assert result["c0"][0].filename == SHOT_NAME


@pytest.mark.asyncio
async def test_a_store_owned_chunk_takes_its_documents_names(monkeypatch):
    store = _memories(store_owned=True, names={"doc-1": {SHOT: SHOT_NAME}})
    monkeypatch.setattr(memories_module, "get_memories", lambda: store)

    result = await MemoryEngine.attachments_for_chunks(
        _EngineWithConn(_NoChunkTableConn()),
        "bank-1",
        ["c0", "c1"],
        request_context=None,
        carried_texts={
            "c0": ("doc-1", f"click {attachment_placeholder(SHOT_HASH)}"),
            "c1": ("doc-1", f"again {attachment_placeholder(SHOT_HASH)}"),
        },
    )

    assert [r.filename for r in result["c0"]] == [SHOT_NAME]
    assert [r.filename for r in result["c1"]] == [SHOT_NAME]
    assert store.record_reads == [], "the name is on the attachment row; the record is not read"


async def _seed_document(store: _CarryingStore, bank_id: str, *, keep_text: bool) -> str:
    text = f"To reset the VPN, click the button: {attachment_placeholder(SHOT_HASH)}\nThen reconnect."
    await store.put_document(
        bank_id=bank_id,
        document_id=DOCUMENT_ID,
        content_hash="h",
        original_text=text if keep_text else None,
        chunk_texts=[text, "Then reconnect."],
        metadata=document_record_metadata(None, {SHOT: SHOT_NAME}),
    )
    return text


@pytest.mark.asyncio
async def test_document_and_chunk_reads_return_the_attachments_in_the_stored_text(
    api_client, memory, request_context, restore_default_store
):
    bank_id, store = await _store_owned_bank(memory, request_context, answers_full_recall=True)

    document = await api_client.get(f"/v1/default/banks/{bank_id}/documents/{DOCUMENT_ID}")
    chunks = await api_client.get(f"/v1/default/banks/{bank_id}/documents/{DOCUMENT_ID}/chunks")
    chunk = await api_client.get(f"/v1/default/chunks/{build_chunk_id(bank_id, DOCUMENT_ID, 0)}")
    # Every one of them takes its attachments, names included, off the attachment rows.
    assert store.record_reads == []

    assert document.status_code == 200, document.text
    _assert_shot(document.json().get("attachments"), bank_id)
    # The carrier is taken off: the payload is the same shape on either backend.
    assert "attachment_filenames" not in document.json()

    assert chunks.status_code == 200, chunks.text
    items = chunks.json()["items"]
    _assert_shot(items[0].get("attachments"), bank_id)
    assert items[1].get("attachments") is None

    assert chunk.status_code == 200, chunk.text
    _assert_shot(chunk.json().get("attachments"), bank_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("keep_text", [True, False], ids=["full-text", "chunks-only"])
async def test_the_retain_revisit_lookup_reads_the_attachment_rows(
    memory, request_context, restore_default_store, keep_text
):
    """The retain ingress asks which attachments a document already carries, with no text in hand,
    so that an edit re-sending its placeholders keeps them. The attachment row names its document,
    so the answer is the same query on either backend -- and it does not depend on whether the
    deployment keeps the document's full text."""
    bank_id, store = await _store_owned_bank(memory, request_context, answers_full_recall=True)
    await _seed_document(store, bank_id, keep_text=keep_text)

    existing = await memory.attachments_for_documents(bank_id, [DOCUMENT_ID, "never-retained"], request_context)

    assert {d: [r.short_id for r in records] for d, records in existing.items()} == {DOCUMENT_ID: [SHOT]}
    # The name rides on the same row, which is what stops an edit from erasing it.
    assert existing[DOCUMENT_ID][0].filename == SHOT_NAME
    assert store.record_reads == []


# -- the write side, through a real retain ----------------------------------

PDF = b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF\n"
PDF_ID = short_attachment_id(compute_attachment_hash(PDF))


def _file_item(document_id: str, filename: str | None) -> dict:
    block = {
        "type": "file",
        "source": {"type": "base64", "media_type": "application/pdf", "data": base64.b64encode(PDF).decode()},
    }
    if filename:
        block["filename"] = filename
    return {"content": [{"type": "text", "text": "See the attached policy."}, block], "document_id": document_id}


@pytest.mark.asyncio
async def test_a_store_owned_retain_writes_the_names_onto_the_document_record(
    api_client, memory, request_context, restore_default_store
):
    """The retain writes the map; the reads return it -- no SQL `document_attachments` row involved."""
    bank_id = f"so-retain-{uuid.uuid4().hex[:8]}"
    store = _CarryingStore(answers_full_recall=True)
    set_memories(store)
    assert (await api_client.put(f"/v1/default/banks/{bank_id}", json={})).status_code == 200
    config = await api_client.patch(
        f"/v1/default/banks/{bank_id}/config", json={"updates": {"retain_extraction_mode": "chunks"}}
    )
    assert config.status_code == 200, config.text

    retained = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={"items": [_file_item("policy", "policy-v1.pdf")], "async": False},
    )
    assert retained.status_code == 200, retained.text

    assert document_attachment_filenames(store.documents["policy"]) == {PDF_ID: "policy-v1.pdf"}
    document = await api_client.get(f"/v1/default/banks/{bank_id}/documents/policy")
    assert document.status_code == 200, document.text
    assert [(a["id"], a["filename"]) for a in document.json()["attachments"]] == [(PDF_ID, "policy-v1.pdf")]


@pytest.mark.asyncio
async def test_a_store_owned_re_ingest_drops_the_attachment_its_text_no_longer_carries(
    api_client, memory, request_context, restore_default_store
):
    """What a document carries is derived from its text, on both backends.

    A bank whose documents live in SQL rewrites the rows from its document write. A store-owned
    bank makes no such write, so the rewrite had nothing to hang off and never ran: the row -- and
    the bytes behind it -- outlived every reference to them, and the document went on reporting an
    attachment its text had stopped carrying.
    """
    bank_id = f"so-reingest-{uuid.uuid4().hex[:8]}"
    store = _CarryingStore(answers_full_recall=True)
    set_memories(store)
    assert (await api_client.put(f"/v1/default/banks/{bank_id}", json={})).status_code == 200
    config = await api_client.patch(
        f"/v1/default/banks/{bank_id}/config", json={"updates": {"retain_extraction_mode": "chunks"}}
    )
    assert config.status_code == 200, config.text

    retained = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={"items": [_file_item("policy", "policy-v1.pdf")], "async": False},
    )
    assert retained.status_code == 200, retained.text
    carried = await api_client.get(f"/v1/default/banks/{bank_id}/documents/policy")
    assert [a["id"] for a in carried.json()["attachments"]] == [PDF_ID]

    re_ingested = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={"items": [{"content": "The policy is now plain text.", "document_id": "policy"}], "async": False},
    )
    assert re_ingested.status_code == 200, re_ingested.text

    document = await api_client.get(f"/v1/default/banks/{bank_id}/documents/policy")
    assert document.status_code == 200, document.text
    assert not document.json().get("attachments")
    # The row too, not only what the read renders: the row is what holds the storage key, so a row
    # left behind is a blob nothing will ever reclaim.
    backend = await memory._get_backend()
    async with backend.acquire() as conn:
        rows = await conn.fetch("SELECT short_id FROM attachments WHERE bank_id = $1", bank_id)
    assert rows == []
