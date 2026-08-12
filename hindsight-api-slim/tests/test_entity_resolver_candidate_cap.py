"""Candidate-set bounding in entity resolution (GH-3211).

Candidate volume per query text is whatever the fuzzy probe returns. On a bank
with many near-identical names that can be thousands of rows, and every one of
them costs a synchronous ``SequenceMatcher`` call on the event-loop thread — a
resolution batch then blocks the worker for minutes, so ``/health`` stops
answering and the orchestrator kills the worker mid-op.

These tests pin the two guarantees that prevent it:
  1. at most ``entity_resolution_max_candidates`` candidates are scored per mention;
  2. the scoring loop yields, so other tasks keep getting scheduled while it runs.
"""

import asyncio
import time
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.entity_resolver import EntityResolver


def _make_resolver(max_candidates: int = 200) -> EntityResolver:
    """Resolver whose entity INSERTs are stubbed (no DB), so only scoring runs."""
    pool = MagicMock()
    ops = MagicMock()
    ops.bulk_insert_entities = AsyncMock(
        side_effect=lambda conn, table, bank_id, names, dates, kinds: {n.lower(): uuid.uuid4() for n in names}
    )
    ops.fetch_missing_entity_ids = AsyncMock(return_value=[])
    pool.ops = ops
    return EntityResolver(
        pool=pool,
        entity_lookup="trigram",
        entity_resolution_max_candidates=max_candidates,
    )  # type: ignore[arg-type]


def _make_conn() -> MagicMock:
    conn = MagicMock()
    conn.backend_type = "postgresql"
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchval = AsyncMock(return_value=True)
    conn.executemany = AsyncMock()
    return conn


def _candidates(count: int, name: str = "Acme Corporation") -> list[tuple]:
    """A polluted candidate set: many distinct names sharing trigrams with the query."""
    now = datetime.now(UTC)
    return [(uuid.uuid4(), f"{name} variant {i:05d}", None, now, 1) for i in range(count)]


@pytest.mark.asyncio
async def test_scoring_is_capped_at_max_candidates():
    """Only the top `max_candidates` per mention reach SequenceMatcher."""
    resolver = _make_resolver(max_candidates=50)
    candidates = _candidates(1000)
    entities_data = [{"text": f"Acme Corporation {i}", "nearby_entities": []} for i in range(4)]
    all_candidates = {e["text"]: candidates for e in entities_data}

    with patch(
        "hindsight_api.engine.entity_resolver.SequenceMatcher",
        wraps=__import__("difflib").SequenceMatcher,
    ) as matcher:
        await resolver._resolve_from_candidates(
            _make_conn(), "bank-1", entities_data, datetime.now(UTC), all_candidates, {}, None, None
        )

    # 4 mentions x 50 candidates — not 4 x 1000.
    assert matcher.call_count == 4 * 50


@pytest.mark.asyncio
async def test_capping_keeps_the_exact_match():
    """Truncation must not drop the candidate that actually matches the mention."""
    resolver = _make_resolver(max_candidates=10)
    exact_id = uuid.uuid4()
    now = datetime.now(UTC)
    # The right answer sits at the very end of a large noisy candidate set.
    candidates = _candidates(500) + [(exact_id, "Acme Corporation", None, now, 1)]
    entities_data = [{"text": "Acme Corporation", "nearby_entities": []}]

    resolved = await resolver._resolve_from_candidates(
        _make_conn(),
        "bank-1",
        entities_data,
        now,
        {"Acme Corporation": candidates},
        {},
        None,
        None,
    )

    assert resolved[0].entity_id == str(exact_id)
    assert resolved[0].canonical_name == "Acme Corporation"


@pytest.mark.asyncio
async def test_scoring_loop_keeps_the_event_loop_responsive():
    """A wide resolution batch must not starve other tasks (e.g. the /health handler).

    Asserts on scheduling, not wall time: a concurrent ticker has to keep running
    while scoring is in flight, and no single stall may approach the whole batch.
    """
    resolver = _make_resolver(max_candidates=2000)
    candidates = _candidates(2000)
    entities_data = [{"text": f"Acme Corporation {i}", "nearby_entities": []} for i in range(20)]
    all_candidates = {e["text"]: candidates for e in entities_data}
    conn = _make_conn()

    # First call in a process lazily imports heavy optional deps; warm that up so
    # it isn't charged to the measured batch.
    await resolver._resolve_from_candidates(
        conn, "bank-1", [{"text": "warm up", "nearby_entities": []}], None, {"warm up": []}, {}, None, None
    )

    stop = False
    gaps: list[float] = []

    async def ticker() -> None:
        last = time.perf_counter()
        while not stop:
            await asyncio.sleep(0.005)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    tick_task = asyncio.create_task(ticker())
    await asyncio.sleep(0.01)

    started = time.perf_counter()
    await resolver._resolve_from_candidates(
        conn, "bank-1", entities_data, datetime.now(UTC), all_candidates, {}, None, None
    )
    elapsed = time.perf_counter() - started

    stop = True
    await tick_task

    during = gaps[1:]
    assert during, "ticker never ran"
    # Generous bounds: this asserts the loop is handed back periodically, not that
    # the machine is fast. Before the fix the ticker got zero turns for the whole
    # batch and the single stall equalled `elapsed`.
    assert max(during) < max(elapsed / 2, 0.5), (
        f"event loop blocked for {max(during):.3f}s during a {elapsed:.3f}s scoring batch"
    )


@pytest.mark.asyncio
async def test_trigram_query_caps_candidates_per_query_text():
    """The PG probe truncates in SQL, pre-ranked by real trigram similarity."""
    resolver = _make_resolver(max_candidates=42)
    conn = _make_conn()

    with patch.object(resolver, "_resolve_from_candidates", new=AsyncMock(return_value=[])):
        await resolver._resolve_entities_batch_trigram(
            conn=conn,
            bank_id="bank-1",
            entities_data=[{"text": "Alice", "nearby_entities": [], "event_date": None}],
            unit_event_date=None,
        )

    query = conn.fetch.call_args.args[0]
    assert "LATERAL" in query
    assert "LIMIT $3" in query
    assert "ORDER BY similarity(" in query
    assert conn.fetch.call_args.args[3] == 42


@pytest.mark.asyncio
async def test_oracle_query_caps_candidates_per_query_text():
    """The Oracle probe truncates per query text via ROW_NUMBER, ranked by Jaro-Winkler."""
    resolver = _make_resolver(max_candidates=42)
    resolver.entity_lookup = "oracle_fuzzy"
    conn = _make_conn()
    conn.backend_type = "oracle"

    with patch.object(resolver, "_resolve_from_candidates", new=AsyncMock(return_value=[])):
        await resolver._resolve_entities_batch_oracle_fuzzy(
            conn=conn,
            bank_id="bank-1",
            entities_data=[{"text": "Alice", "nearby_entities": [], "event_date": None}],
            unit_event_date=None,
        )

    query = conn.fetch.call_args.args[0]
    assert "ROW_NUMBER() OVER" in query
    assert "PARTITION BY q.query_text" in query
    assert "WHERE rn <= $3" in query
    assert conn.fetch.call_args.args[3] == 42


def test_max_candidates_must_be_positive():
    with pytest.raises(ValueError, match="entity_resolution_max_candidates must be >= 1"):
        EntityResolver(pool=None, entity_resolution_max_candidates=0)  # type: ignore[arg-type]
