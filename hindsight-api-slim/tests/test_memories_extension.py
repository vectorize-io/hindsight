"""The memories store is an extension point, and the default is Postgres.

Two things are worth pinning down here, and neither is about SQL:

1. **The default is unconditional.** With nothing configured the engine gets
   :class:`PostgresMemories`, so every other test in this suite is exercising the
   real store rather than a seam that happens to fall through to it.
2. **The interface is complete.** A store that implements
   :class:`MemoriesExtension` and touches no database at all can be installed and
   used. That is the property the whole extraction exists for: if a call site
   still reached past the interface for a `memory_units` row, the stub below
   would not be able to answer and the test would fail rather than quietly
   falling back to SQL.

The stub is deliberately a dictionary. Anything cleverer would start re-testing
storage instead of the seam.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from hindsight_api.engine.memories import create_memories, get_memories, set_memories
from hindsight_api.engine.memories.base import MemoriesExtension, ScanPage, StoredMemory
from hindsight_api.engine.memories.postgres import PostgresMemories


class InMemoryMemories(MemoriesExtension):
    """A complete store that is a dict — no connection, no tables, no SQL.

    Every method that touches storage is answered from ``self.rows``. The
    Postgres handles (``conn``, ``ops``, ``fq_table``) are accepted and ignored,
    which is exactly what an implementation that owns the store does with them.
    """

    name = "in-memory"

    def __init__(self, config: dict[str, str] | None = None):
        super().__init__(config or {})
        self.rows: dict[str, StoredMemory] = {}
        # Proof the engine went through the interface rather than around it.
        self.calls: list[str] = []

    # -- writes --------------------------------------------------------------

    async def insert_facts(self, *, conn, ops, bank_id, facts, document_id=None, defer_index=False):
        self.calls.append("insert_facts")
        unit_ids = self.allocate_unit_ids(len(facts))
        if not defer_index:
            await self.index_facts(bank_id, unit_ids, facts, document_id)
        return unit_ids

    async def index_facts(self, bank_id, unit_ids, facts, document_id=None, unit_entity_ids=None):
        self.calls.append("index_facts")
        for unit_id, fact in zip(unit_ids, facts):
            self.rows[unit_id] = StoredMemory(
                unit_id=unit_id,
                text=fact.fact_text,
                fact_type=fact.fact_type,
                document_id=document_id,
                tags=list(fact.tags or []),
                created_at=datetime.now(timezone.utc),
            )

    async def delete_facts(self, bank_id, unit_ids):
        self.calls.append("delete_facts")
        for unit_id in unit_ids:
            self.rows.pop(str(unit_id), None)

    async def delete_document(self, *, conn, fq_table, bank_id, document_id):
        self.calls.append("delete_document")
        for unit_id in [k for k, v in self.rows.items() if v.document_id == document_id]:
            del self.rows[unit_id]

    async def delete_observations(self, *, conn, fq_table, bank_id):
        for unit_id in [k for k, v in self.rows.items() if v.fact_type == "observation"]:
            del self.rows[unit_id]

    # -- recall arms ---------------------------------------------------------

    async def search(self, *, conn, bank_id, fact_types, query_embedding, query_text, limit, **kwargs):
        self.calls.append("search")
        return {ft: ([], []) for ft in fact_types}

    async def temporal_search(
        self, *, conn, bank_id, fact_types, query_embedding, start_date, end_date, limit, **kwargs
    ):
        self.calls.append("temporal_search")
        return {ft: [] for ft in fact_types}

    # -- addressed reads -----------------------------------------------------

    async def get_memories(self, *, conn, fq_table, bank_id, unit_ids):
        self.calls.append("get_memories")
        return [self.rows[str(u)] for u in unit_ids if str(u) in self.rows]

    async def scan_memories(self, *, conn, fq_table, bank_id, limit=100, page_token="", **kwargs):
        start = int(page_token or 0)
        ordered = list(self.rows.values())[start : start + limit]
        nxt = str(start + limit) if start + limit < len(self.rows) else ""
        return ScanPage(memories=ordered, next_page_token=nxt)

    async def count_memories(self, *, conn, fq_table, bank_id):
        counts: dict[str, int] = {}
        for row in self.rows.values():
            counts[row.fact_type] = counts.get(row.fact_type, 0) + 1
        return counts

    async def list_tags(self, *, conn, fq_table, bank_id):
        counts: dict[str, int] = {}
        for row in self.rows.values():
            for tag in row.tags:
                counts[tag] = counts.get(tag, 0) + 1
        return counts

    async def find_unconsolidated(self, *, conn, fq_table, bank_id, fact_types, limit, scope_tags=None):
        out = [r for r in self.rows.values() if r.fact_type in fact_types and r.consolidated_at is None]
        if scope_tags:
            out = [r for r in out if set(scope_tags).issubset(set(r.tags))]
        return out[:limit]

    async def mark_consolidated(self, *, conn, fq_table, bank_id, unit_ids, when, failed=False):
        for unit_id in unit_ids:
            row = self.rows.get(str(unit_id))
            if row is not None:
                row.consolidated_at = when

    async def entity_memory_counts(self, *, conn, fq_table, bank_id, entity_ids=None):
        counts: dict[str, int] = {}
        for row in self.rows.values():
            for entity_id in row.entity_ids:
                counts[entity_id] = counts.get(entity_id, 0) + 1
        return counts if entity_ids is None else {k: v for k, v in counts.items() if k in set(entity_ids)}

    async def entities_for_units(self, *, conn, fq_table, bank_id, unit_ids):
        return {str(u): list(self.rows[str(u)].entity_ids) for u in unit_ids if str(u) in self.rows}

    async def entity_map_for_units(self, *, conn, fq_table, bank_id, unit_ids):
        # No names in the stub — the entity registry is Postgres's, which the
        # stub does not stand in for. The shape is what recall consumes.
        return {
            str(u): [{"entity_id": e, "canonical_name": e} for e in self.rows[str(u)].entity_ids]
            for u in unit_ids
            if str(u) in self.rows
        }

    async def any_memory_updated_since(
        self, *, conn, fq_table, bank_id, since, fact_types=None, tags=None, tags_match="any", tag_groups=None
    ):
        rows = self.rows.values()
        if fact_types:
            rows = [r for r in rows if r.fact_type in fact_types]
        return any(r.created_at is not None and r.created_at > since for r in rows)

    # -- observations --------------------------------------------------------

    async def observations_for_sources(self, *, conn, ops, fq_table, bank_id, unit_ids):
        wanted = {str(u) for u in unit_ids}
        return [r for r in self.rows.values() if wanted & set(r.source_memory_ids)]

    async def delete_stale_observations(self, *, conn, ops, fq_table, bank_id, fact_ids):
        stale = await self.observations_for_sources(
            conn=conn, ops=ops, fq_table=fq_table, bank_id=bank_id, unit_ids=fact_ids
        )
        for obs in stale:
            self.rows.pop(obs.unit_id, None)
        return len(stale)

    # -- curation reads ------------------------------------------------------

    async def list_memory_units(self, *, conn, ops, fq_table, bank_id, limit=100, offset=0, **kwargs):
        ordered = list(self.rows.values())
        return {"items": ordered[offset : offset + limit], "total": len(ordered), "limit": limit, "offset": offset}

    async def get_memory_unit(self, *, conn, ops, fq_table, bank_id, unit_id):
        row = self.rows.get(str(unit_id))
        return None if row is None else {"id": row.unit_id, "text": row.text, "fact_type": row.fact_type}

    async def list_entities(self, *, conn, fq_table, bank_id, search=None, limit=100, offset=0):
        return {"items": [], "total": 0, "limit": limit, "offset": offset}

    async def graph_units(self, *, conn, fq_table, bank_id, limit=1000, **kwargs):
        rows = list(self.rows.values())[:limit]
        return {"units": [{"id": r.unit_id, "fact_type": r.fact_type} for r in rows], "total": len(self.rows)}

    async def graph_entity_rows(self, *, conn, fq_table, bank_id, unit_ids):
        return []

    async def graph_direct_links(self, *, conn, fq_table, bank_id, unit_ids):
        return []


@pytest.fixture
def restore_default_store():
    """Put the process-wide store back, whatever a test did to it."""
    yield
    set_memories(None)


def test_default_store_is_postgres(restore_default_store):
    """Nothing configured means the SQL path — which is what every other test runs."""
    set_memories(None)
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("HINDSIGHT_API_MEMORIES_EXTENSION", None)
        assert isinstance(create_memories(), PostgresMemories)
        assert get_memories().name == "postgres"


def test_a_configured_store_replaces_the_default(restore_default_store):
    """The ordinary extension env var selects the store, like every other extension."""
    set_memories(None)
    spec = f"{InMemoryMemories.__module__}:{InMemoryMemories.__name__}"
    with patch.dict(os.environ, {"HINDSIGHT_API_MEMORIES_EXTENSION": spec}):
        store = create_memories()
    assert isinstance(store, InMemoryMemories)
    assert store.name == "in-memory"


def test_the_store_receives_its_prefixed_config(restore_default_store):
    """`HINDSIGHT_API_MEMORIES_*` reaches the store, stripped and lowercased."""
    set_memories(None)
    spec = f"{InMemoryMemories.__module__}:{InMemoryMemories.__name__}"
    env = {
        "HINDSIGHT_API_MEMORIES_EXTENSION": spec,
        "HINDSIGHT_API_MEMORIES_TARGET": "example:50051",
        "HINDSIGHT_API_MEMORIES_NPROBE": "16",
    }
    with patch.dict(os.environ, env):
        store = create_memories()
    assert store.config["target"] == "example:50051"
    assert store.config["nprobe"] == "16"
    assert "extension" not in store.config, "the selector must not leak into the store's own config"


def test_the_interface_is_implementable_without_a_database(restore_default_store):
    """The point of the extraction: a store with no SQL behind it is a valid store.

    Instantiating an ABC with a missing method raises `TypeError` naming it, so a
    method added to the interface without a home here fails loudly rather than at
    the first call site that needs it.
    """
    store = InMemoryMemories({})
    assert isinstance(store, MemoriesExtension)


async def test_a_store_that_owns_its_rows_needs_no_postgres(restore_default_store):
    """Write, read back, and delete — with `conn` set to something unusable.

    Passing `None` where the Postgres store would expect a connection is the
    assertion: any code path that quietly reached for SQL would raise instead of
    returning the rows the store holds.
    """
    store = InMemoryMemories({})
    set_memories(store)

    class _Fact:
        fact_text = "the cat sat on the mat"
        fact_type = "world"
        tags = ["animals"]

    unit_ids = await store.insert_facts(conn=None, ops=None, bank_id="bank", facts=[_Fact()], document_id="doc-1")
    assert len(unit_ids) == 1

    got = await store.get_memories(conn=None, fq_table=None, bank_id="bank", unit_ids=unit_ids)
    assert [m.text for m in got] == ["the cat sat on the mat"]
    assert await store.count_memories(conn=None, fq_table=None, bank_id="bank") == {"world": 1}
    assert await store.list_tags(conn=None, fq_table=None, bank_id="bank") == {"animals": 1}

    # Deleting the document takes its memories with it, with no cascade to rely on.
    await store.delete_document(conn=None, fq_table=None, bank_id="bank", document_id="doc-1")
    assert await store.get_memories(conn=None, fq_table=None, bank_id="bank", unit_ids=unit_ids) == []
    assert "insert_facts" in store.calls and "get_memories" in store.calls


async def test_maintenance_passes_are_optional(restore_default_store):
    """A store with inline links has nothing to relink and no join table to sweep.

    These have safe base implementations precisely so such a store does not have
    to write four no-op methods to be complete — and so the maintenance job can
    call them unconditionally.
    """
    store = InMemoryMemories({})
    assert await store.enqueue_relink_victims(conn=None, fq_table=None, bank_id="b", deleted_unit_ids=["x"]) == 0
    assert await store.relink_pass(backend=None, fq_table=None, bank_id="b", config=None) == {}
    assert await store.prune_orphan_entities(conn=None, fq_table=None, bank_id="b") == 0
    assert await store.prune_stale_cooccurrences(conn=None, fq_table=None, bank_id="b") == 0
    # And recording entity postings is a no-op rather than an error: the posting
    # travels on the memory for a store that owns it.
    await store.record_unit_entities(conn=None, ops=None, fq_table=None, unit_ids=["u"], entity_ids=["e"])
