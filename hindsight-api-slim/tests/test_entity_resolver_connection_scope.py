"""Entity resolution holds no pool connection while it computes in Python (#6603).

Retain's phase 1 used to take one connection and keep it across the whole resolution:
the candidate reads, the co-occurrence index, the scoring loop, the in-batch clustering and
the entity INSERTs. Under concurrent retain each running batch then pinned a connection for
the length of a Python loop, and ``/health``'s ``SELECT 1`` queued behind them. These tests
pin the split: a connection is held only around each database call.
"""

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.engine import db_utils
from hindsight_api.engine.memories.pg import entity_resolver as entity_resolver_module
from hindsight_api.engine.memories.pg.entity_resolver import EntityResolver
from hindsight_api.engine.retain import entity_processing, orchestrator
from hindsight_api.engine.retain.orchestrator import _pre_resolve_phase1
from hindsight_api.engine.retain.types import EntityResolutionResult, ProcessedFact, RetainContent

NOW = datetime.now(UTC)


class _HeldConnections:
    """Counts the connections acquired and not yet released."""

    def __init__(self, conn) -> None:
        self.conn = conn
        self.held = 0
        self.acquired = 0

    @asynccontextmanager
    async def acquire(self, _pool):
        self.held += 1
        self.acquired += 1
        try:
            yield self.conn
        finally:
            self.held -= 1


def _candidate_rows(query: str, count: int) -> list[dict]:
    return [
        {
            "id": f"00000000-0000-0000-0000-{i:012d}",
            "canonical_name": f"{query} {i}",
            "metadata": None,
            "last_seen": NOW,
            "mention_count": 1,
            "query_text": query,
        }
        for i in range(count)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("entity_lookup", "expected_holds"),
    [
        ("trigram", 4),  # pg_trgm check, candidate probes, co-occurrences, entity INSERT
        ("full", 2),  # every bank entity + co-occurrence in one read, entity INSERT
    ],
)
async def test_resolution_computes_with_no_connection_held(monkeypatch, entity_lookup, expected_holds):
    conn = MagicMock()
    conn.backend_type = "postgresql"
    conn.fetchval = AsyncMock(return_value=True)  # pg_trgm installed

    async def fetch(sql, *args):
        if "entity_cooccurrences" in sql:
            return [{"entity_id_1": "00000000-0000-0000-0000-000000000000", "entity_id_2": "x"}]
        # Only "Acme Corporation" has look-alikes in the bank, so it is scored and the other
        # name takes the INSERT path. "full" passes no texts: it loads the whole bank.
        texts = args[1] if len(args) > 1 else ["Acme Corporation"]
        return [row for text in texts if text == "Acme Corporation" for row in _candidate_rows(text, 20)]

    conn.fetch = AsyncMock(side_effect=fetch)
    counter = _HeldConnections(conn)
    monkeypatch.setattr(db_utils, "acquire_with_retry", counter.acquire)
    monkeypatch.setattr(entity_resolver_module, "acquire_with_retry", counter.acquire)

    held_while_computing: list[int] = []

    def spy(fn):
        def wrapped(*args, **kwargs):
            held_while_computing.append(counter.held)
            return fn(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(
        entity_resolver_module, "_build_cooccurrence_index", spy(entity_resolver_module._build_cooccurrence_index)
    )
    # The scoring loop's per-candidate gate, and the in-batch clustering.
    monkeypatch.setattr(
        entity_resolver_module, "trigram_set_similarity", spy(entity_resolver_module.trigram_set_similarity)
    )
    monkeypatch.setattr(EntityResolver, "_intrabatch_canonical_map", spy(EntityResolver._intrabatch_canonical_map))

    pool = MagicMock()
    pool.ops = MagicMock()

    async def bulk_insert(conn, table, bank_id, names, dates, kinds):
        assert counter.held == 1, "the entity INSERT runs on a held connection"
        return {n.lower(): f"new-{n}" for n in names}

    pool.ops.bulk_insert_entities = AsyncMock(side_effect=bulk_insert)
    pool.ops.fetch_missing_entity_ids = AsyncMock(return_value=[])
    resolver = EntityResolver(pool=pool, entity_lookup=entity_lookup)  # type: ignore[arg-type]

    resolved = await resolver.resolve_entities_batch(
        bank_id="bank",
        entities_data=[
            {"text": "Acme Corporation", "nearby_entities": [{"text": "Globex"}]},
            {"text": "Brand New Thing", "nearby_entities": []},
        ],
        context="",
        unit_event_date=NOW,
        conn=None,
    )

    assert len(resolved) == 2
    assert held_while_computing, "the spies must have seen the compute steps run"
    assert set(held_while_computing) == {0}, "no connection may be held while resolution computes"
    assert counter.held == 0
    assert counter.acquired == expected_holds


@pytest.mark.asyncio
async def test_phase1_does_not_hold_a_connection_across_resolution(monkeypatch):
    """Phase 1 hands resolution the pool, not a connection it keeps open meanwhile."""
    counter = _HeldConnections(object())
    monkeypatch.setattr(orchestrator, "acquire_with_retry", counter.acquire)
    seen: list[tuple] = []

    async def fake_resolve_entities(_resolver, conn, *_args, **_kwargs):
        seen.append((conn, counter.held))
        return EntityResolutionResult(resolved_entities=[], entity_to_unit=[], unit_to_entity_ids={})

    monkeypatch.setattr(entity_processing, "resolve_entities", fake_resolve_entities)

    await _pre_resolve_phase1(
        pool=object(),
        entity_resolver=object(),
        bank_id="bank",
        contents=[RetainContent(content="Alice works at Acme")],
        processed_facts=[
            ProcessedFact(
                fact_text="Alice works at Acme",
                fact_type="world",
                embedding=[0.1],
                occurred_start=None,
                occurred_end=None,
                mentioned_at=NOW,
                context="",
                metadata={},
            )
        ],
        config=SimpleNamespace(entity_labels=None, semantic_link_min_similarity=0.8),
        log_buffer=[],
        skip_semantic_ann=True,
    )

    assert seen == [(None, 0)]
