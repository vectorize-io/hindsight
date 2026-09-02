"""Single-fetch fusion of the observation graph arm (issue #3857).

``expand_observations`` used to run the entity/source traversal and the
semantic/causal expansion as two separate statements. The entity arm is now
fused into the semantic/causal CTE query behind a ``source`` discriminator
(the same shape as the non-observation combined expansion), so a normal call
performs one fetch instead of two. These tests pin what the fusion must
preserve: identical ordered IDs, scores, and counts across overlapping
sources, duplicate entities, empty sources, and multiple fact types; time
windows binding every arm; a source-less seed still getting its semantic and
causal neighbours; the fetch count itself on a live backend; and the
one-statement query shape for both PostgreSQL and Oracle without a live
Oracle instance.
"""

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from hindsight_api.engine.db.ops import UpdatedWindow
from hindsight_api.engine.db.ops_oracle import OracleOps
from hindsight_api.engine.db.ops_postgresql import PostgreSQLOps


class _CountingConn:
    """Pass-through connection wrapper that counts ``fetch`` calls."""

    def __init__(self, conn):
        self._conn = conn
        self.fetch_calls = 0

    async def fetch(self, query, *args):
        self.fetch_calls += 1
        return await self._conn.fetch(query, *args)

    def __getattr__(self, name):
        return getattr(self._conn, name)


async def _ensure_bank(conn, bank_id: str) -> None:
    await conn.execute(
        "INSERT INTO banks (bank_id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        bank_id,
        bank_id,
    )


async def _insert_unit(
    conn,
    table: str,
    bank_id: str,
    text: str,
    fact_type: str,
    sources: list[uuid.UUID] | None = None,
) -> uuid.UUID:
    unit_id = uuid.uuid4()
    await conn.execute(
        f"""
        INSERT INTO {table} (id, bank_id, text, fact_type, source_memory_ids, event_date)
        VALUES ($1, $2, $3, $4, $5::uuid[], $6)
        """,
        unit_id,
        bank_id,
        text,
        fact_type,
        sources,
        datetime.now(timezone.utc),
    )
    return unit_id


async def _insert_entity(conn, table: str, bank_id: str, name: str) -> uuid.UUID:
    entity_id = uuid.uuid4()
    await conn.execute(
        f"INSERT INTO {table} (id, bank_id, canonical_name) VALUES ($1, $2, $3)",
        entity_id,
        bank_id,
        name,
    )
    return entity_id


async def _insert_link(
    conn, table: str, bank_id: str, from_unit: uuid.UUID, to_unit: uuid.UUID, link_type: str, weight: float
):
    await conn.execute(
        f"INSERT INTO {table} (bank_id, from_unit_id, to_unit_id, link_type, weight) VALUES ($1, $2, $3, $4, $5)",
        bank_id,
        from_unit,
        to_unit,
        link_type,
        weight,
    )


async def _build_overlap_bank(conn, mu, ue, ml, bank_id: str, entities_table: str) -> dict:
    """A bank exercising every fusion-preserved semantic at once.

    Five world facts share one entity, so the seeds' connected-source set is
    {f2, f3, f4, f5} (f1 is a seed source and excluded). On top of it:

    - c3/c1 share overlapping (not identical) subsets of that set → scores 3/1.
    - cdup lists a source twice → still counts once.
    - cempty has no sources → never an entity candidate.
    - two seeds reach the same source fact through the same entity → the
      DISTINCT seed traversal must not double-count it.
    - a world fact carrying a high-weight semantic link to a seed must stay
      excluded by the fact-type filter.
    - sem/caus are observation neighbours reachable only through semantic and
      causal links, with multiple links whose best weight must win.
    """
    entity_id = await _insert_entity(conn, entities_table, bank_id, "Acme")
    f1 = await _insert_unit(conn, mu, bank_id, "source fact 1", "world")
    f2 = await _insert_unit(conn, mu, bank_id, "source fact 2", "world")
    f3 = await _insert_unit(conn, mu, bank_id, "source fact 3", "world")
    f4 = await _insert_unit(conn, mu, bank_id, "source fact 4", "world")
    f5 = await _insert_unit(conn, mu, bank_id, "source fact 5", "world")
    for fid in (f1, f2, f3, f4, f5):
        await conn.execute(f"INSERT INTO {ue} (unit_id, entity_id) VALUES ($1, $2)", fid, entity_id)

    s1 = await _insert_unit(conn, mu, bank_id, "seed one", "observation", [f1])
    s2 = await _insert_unit(conn, mu, bank_id, "seed two", "observation", [f1])
    c3 = await _insert_unit(conn, mu, bank_id, "shares three", "observation", [f2, f3, f4])
    c1 = await _insert_unit(conn, mu, bank_id, "shares one", "observation", [f2])
    cdup = await _insert_unit(conn, mu, bank_id, "duplicate source listed twice", "observation", [f3, f3])
    cempty = await _insert_unit(conn, mu, bank_id, "no sources", "observation", None)

    sem = await _insert_unit(conn, mu, bank_id, "semantic neighbour", "observation")
    caus = await _insert_unit(conn, mu, bank_id, "causal neighbour", "observation")
    await _insert_link(conn, ml, bank_id, s1, sem, "semantic", 0.5)
    await _insert_link(conn, ml, bank_id, s2, sem, "semantic", 0.8)
    await _insert_link(conn, ml, bank_id, s1, caus, "causes", 0.9)
    await _insert_link(conn, ml, bank_id, s1, caus, "enables", 0.6)
    await _insert_link(conn, ml, bank_id, s2, caus, "causes", 0.7)
    # A world fact must never come back, however strongly it is linked.
    await _insert_link(conn, ml, bank_id, s1, f2, "semantic", 0.95)

    return {
        "s1": s1,
        "s2": s2,
        "c3": c3,
        "c1": c1,
        "cdup": cdup,
        "cempty": cempty,
        "sem": sem,
        "caus": caus,
        "world": f2,
    }


@pytest.mark.asyncio
async def test_observation_expansion_performs_one_fetch(memory, request_context):
    """A normal observation expansion performs one fetch, not two (#3857).

    The single fetch must still carry all three arms: the entity/source
    traversal plus the semantic and causal neighbours.
    """
    from hindsight_api.engine.task_backend import fq_table

    bank_id = f"test_obs_one_fetch_{uuid.uuid4().hex[:8]}"
    try:
        pool = await memory._get_pool()
        backend = await memory._get_backend()
        mu, ue, ml = fq_table("memory_units"), fq_table("unit_entities"), fq_table("memory_links")
        entities_table = fq_table("entities")

        async with pool.acquire() as conn:
            await _ensure_bank(conn, bank_id)
            ids = await _build_overlap_bank(conn, mu, ue, ml, bank_id, entities_table)

            counting = _CountingConn(conn)
            rows = await backend.ops.expand_observations(
                counting,
                mu,
                ue,
                ml,
                [ids["s1"], ids["s2"]],
                100,
                200,
                UpdatedWindow(after=None, before=None, first_param_index=3),
            )

        assert counting.fetch_calls == 1, "the fused observation expansion must fetch exactly once"
        assert {r["id"] for r in rows.entity} == {ids["c3"], ids["c1"], ids["cdup"]}
        assert {r["id"] for r in rows.semantic} == {ids["sem"]}
        assert {r["id"] for r in rows.causal} == {ids["caus"]}
    finally:
        await memory.delete_bank(bank_id=bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_fusion_preserves_ids_scores_counts_and_ordering(memory, request_context):
    """IDs, scores, counts, and ordering survive the fusion unchanged.

    Score = number of distinct shared source facts (duplicates count once,
    entities shared by several seeds are not double-counted); ordering is by
    score descending; each arm keeps its own budget limit.
    """
    from hindsight_api.engine.task_backend import fq_table

    bank_id = f"test_obs_fused_{uuid.uuid4().hex[:8]}"
    try:
        pool = await memory._get_pool()
        backend = await memory._get_backend()
        mu, ue, ml = fq_table("memory_units"), fq_table("unit_entities"), fq_table("memory_links")
        entities_table = fq_table("entities")

        async with pool.acquire() as conn:
            await _ensure_bank(conn, bank_id)
            ids = await _build_overlap_bank(conn, mu, ue, ml, bank_id, entities_table)

            rows = await backend.ops.expand_observations(
                conn,
                mu,
                ue,
                ml,
                [ids["s1"], ids["s2"]],
                100,
                200,
                UpdatedWindow(after=None, before=None, first_param_index=3),
            )

            scores = {r["id"]: r["score"] for r in rows.entity}
            assert scores == {ids["c3"]: 3.0, ids["c1"]: 1.0, ids["cdup"]: 1.0}, (
                "overlapping sources score by distinct shared sources; duplicate ids count once"
            )
            assert len(rows.entity) == 3
            ranked = [r["id"] for r in rows.entity]
            assert ranked[0] == ids["c3"], "higher shared-source count must rank first"

            # The seed's own id never comes back, and empty-source observations
            # are not entity candidates.
            for absent in (ids["s1"], ids["s2"], ids["cempty"]):
                assert absent not in scores

            # Semantic: MAX(weight) across multiple links from the seed set;
            # the world fact linked with weight 0.95 stays filtered out.
            assert {r["id"]: r["score"] for r in rows.semantic} == {ids["sem"]: 0.8}
            # Causal: best weight across all causal-type links to the seed set.
            assert {r["id"]: r["score"] for r in rows.causal} == {ids["caus"]: 0.9}

            # Each arm keeps its own budget limit, ordering intact: budget 2
            # caps the entity arm at its top two rows but leaves the one-row
            # semantic and causal arms intact (a *global* budget would cut
            # them to zero).
            limited = await backend.ops.expand_observations(
                conn,
                mu,
                ue,
                ml,
                [ids["s1"], ids["s2"]],
                2,
                200,
                UpdatedWindow(after=None, before=None, first_param_index=3),
            )
            assert len(limited.entity) == 2
            assert [r["id"] for r in limited.entity][0] == ids["c3"]
            assert len(limited.semantic) == 1
            assert len(limited.causal) == 1
    finally:
        await memory.delete_bank(bank_id=bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_time_window_binds_every_fused_arm(memory, request_context):
    """created_after/created_before narrow the fused arms alike ($3/$4 binds).

    The window bounds the observations that come *back*; in the fused query
    both placeholder positions must bind correctly for every arm — the arms
    share one param list, so a mis-numbered window would fail everywhere.
    """
    from hindsight_api.engine.task_backend import fq_table

    bank_id = f"test_obs_window_{uuid.uuid4().hex[:8]}"
    try:
        pool = await memory._get_pool()
        backend = await memory._get_backend()
        mu, ue, ml = fq_table("memory_units"), fq_table("unit_entities"), fq_table("memory_links")
        entities_table = fq_table("entities")

        async with pool.acquire() as conn:
            await _ensure_bank(conn, bank_id)
            entity_id = await _insert_entity(conn, entities_table, bank_id, "Globex")
            f1 = await _insert_unit(conn, mu, bank_id, "window source", "world")
            f2 = await _insert_unit(conn, mu, bank_id, "window other", "world")
            for fid in (f1, f2):
                await conn.execute(f"INSERT INTO {ue} (unit_id, entity_id) VALUES ($1, $2)", fid, entity_id)

            seed = await _insert_unit(conn, mu, bank_id, "window seed", "observation", [f1])
            fresh = await _insert_unit(conn, mu, bank_id, "fresh observation", "observation", [f2])
            stale = await _insert_unit(conn, mu, bank_id, "stale observation", "observation", [f2])
            await conn.execute(
                f"UPDATE {mu} SET updated_at = $2 WHERE id = $1",
                stale,
                datetime.now(timezone.utc) - timedelta(days=30),
            )
            await _insert_link(conn, ml, bank_id, seed, fresh, "semantic", 0.7)
            await _insert_link(conn, ml, bank_id, seed, stale, "semantic", 0.7)
            await _insert_link(conn, ml, bank_id, seed, fresh, "causes", 0.6)
            await _insert_link(conn, ml, bank_id, seed, stale, "causes", 0.9)

            now = datetime.now(timezone.utc)
            window = UpdatedWindow(
                after=now - timedelta(minutes=5), before=now + timedelta(minutes=5), first_param_index=3
            )
            rows = await backend.ops.expand_observations(conn, mu, ue, ml, [seed], 100, 200, window)

            assert {r["id"] for r in rows.entity} == {fresh}, "stale observations must not re-enter via shared sources"
            assert {r["id"] for r in rows.semantic} == {fresh}, "the window must bind the semantic arm too"
            assert {r["id"] for r in rows.causal} == {fresh}, "the window must bind the causal arm too"

            # Without the window both come back, at identical scores.
            unbounded = await backend.ops.expand_observations(
                conn,
                mu,
                ue,
                ml,
                [seed],
                100,
                200,
                UpdatedWindow(after=None, before=None, first_param_index=3),
            )
            assert {r["id"] for r in unbounded.entity} == {fresh, stale}
            assert {r["id"] for r in unbounded.semantic} == {fresh, stale}
            assert {r["id"] for r in unbounded.causal} == {fresh, stale}
    finally:
        await memory.delete_bank(bank_id=bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_seed_without_sources_keeps_semantic_and_causal_arms(memory, request_context):
    """A seed with no source facts yields an empty entity arm, and that is all.

    The fusion must not make the semantic/causal arms depend on the
    entity/source traversal: with ``source_memory_ids`` NULL the seed-source
    CTE is empty, so the whole entity half of the fused query finds nothing —
    the semantic and causal neighbours must still come back.
    """
    from hindsight_api.engine.task_backend import fq_table

    bank_id = f"test_obs_nosrc_{uuid.uuid4().hex[:8]}"
    try:
        pool = await memory._get_pool()
        backend = await memory._get_backend()
        mu, ue, ml = fq_table("memory_units"), fq_table("unit_entities"), fq_table("memory_links")

        async with pool.acquire() as conn:
            await _ensure_bank(conn, bank_id)
            seed = await _insert_unit(conn, mu, bank_id, "source-less seed", "observation", None)
            sem = await _insert_unit(conn, mu, bank_id, "semantic neighbour", "observation")
            caus = await _insert_unit(conn, mu, bank_id, "causal neighbour", "observation")
            await _insert_link(conn, ml, bank_id, seed, sem, "semantic", 0.8)
            await _insert_link(conn, ml, bank_id, seed, caus, "causes", 0.6)

            rows = await backend.ops.expand_observations(
                conn,
                mu,
                ue,
                ml,
                [seed],
                100,
                200,
                UpdatedWindow(after=None, before=None, first_param_index=3),
            )

        assert list(rows.entity) == [], "a source-less seed has no entity neighbourhood"
        assert {r["id"] for r in rows.semantic} == {sem}
        assert {r["id"] for r in rows.causal} == {caus}
    finally:
        await memory.delete_bank(bank_id=bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_postgresql_fused_query_is_one_statement():
    """Structural (no live DB): PG emits ONE statement carrying all three arms.

    Pins the #3857 shape against accidental re-splitting: a single ``fetch``
    whose SQL ends in the ``observation_entity_expanded UNION ALL
    semantic_expanded UNION ALL causal_expanded`` select, with the bind order
    seeds → budget → window bounds ($3/$4) and a per-arm ``LIMIT $2``.
    """
    conn = AsyncMock()
    conn.fetch.return_value = []

    seeds = [uuid.uuid4(), uuid.uuid4()]
    after, before = datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 2, 1, tzinfo=timezone.utc)
    await PostgreSQLOps().expand_observations(
        conn,
        "memory_units",
        "unit_entities",
        "memory_links",
        seeds,
        100,
        200,
        UpdatedWindow(after=after, before=before, first_param_index=3),
    )

    assert conn.fetch.await_count == 1, "the fused observation expansion must be one fetch"
    sql = conn.fetch.await_args.args[0]
    params = conn.fetch.await_args.args[1:]
    normalized_sql = " ".join(sql.split())

    # One statement, ending in the three-arm union behind source discriminators.
    assert ";" not in sql, "the fused query must be a single statement"
    assert (
        "SELECT * FROM observation_entity_expanded UNION ALL SELECT * FROM semantic_expanded UNION ALL SELECT * FROM causal_expanded"
        in normalized_sql
    )
    for discriminator in ("'entity'::text AS source", "'semantic'::text AS source", "'causal'::text AS source"):
        assert discriminator in sql
    # The entity arm is a CTE of the same statement, not a separate fetch.
    assert sql.count("LIMIT $2") == 3, "each arm keeps its own budget limit"
    assert "updated_at > $3" in sql and "updated_at < $4" in sql, "the window binds at $3/$4 in every arm"
    # Bind order: seeds, budget, then the window bounds.
    assert params == (seeds, 100, after, before)
    assert sql.count("$1::uuid[]") >= 4, "the seed bind reaches every arm"


@pytest.mark.asyncio
async def test_oracle_fused_query_is_one_statement():
    """Structural (no Oracle runtime): Oracle emits ONE statement, three arms.

    Oracle instances are not available to every test run, so the fused shape
    is pinned the way test_db_abstraction pins Oracle behaviour — by invoking
    ``OracleOps.expand_observations`` against a mock connection and inspecting
    the statement and binds it produces.
    """
    conn = AsyncMock()
    conn.fetch.return_value = []

    seeds = [uuid.uuid4()]
    after, before = datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 2, 1, tzinfo=timezone.utc)
    await OracleOps().expand_observations(
        conn,
        "memory_units",
        "unit_entities",
        "memory_links",
        seeds,
        100,
        200,
        UpdatedWindow(after=after, before=before, first_param_index=3),
    )

    assert conn.fetch.await_count == 1, "the fused observation expansion must be one fetch"
    sql = conn.fetch.await_args.args[0]
    params = conn.fetch.await_args.args[1:]
    normalized_sql = " ".join(sql.split())

    # One statement, ending in the three-arm union behind source discriminators.
    assert ";" not in sql, "the fused query must be a single statement"
    assert (
        "SELECT * FROM observation_entity_expanded UNION ALL SELECT * FROM semantic_expanded UNION ALL SELECT * FROM causal_expanded"
        in normalized_sql
    )
    for discriminator in ("'entity' AS source", "'semantic' AS source", "'causal' AS source"):
        assert discriminator in sql
    # The entity arm is a CTE of the same statement, not a separate fetch.
    assert sql.count("FETCH FIRST $2 ROWS ONLY") == 3, "each arm keeps its own budget limit"
    assert "updated_at > $3" in sql and "updated_at < $4" in sql, "the window binds at $3/$4 in every arm"
    # Bind order: seeds, budget, then the window bounds.
    assert params == (seeds, 100, after, before)
    assert sql.count("$1::uuid[]") >= 4, "the seed bind reaches every arm"
