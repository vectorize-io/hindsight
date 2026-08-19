"""Shared-source scoring for the observation graph arm (issue #3085).

``expand_observations`` ranks candidate observations by how many source facts
they share with the seeds' entity neighbourhood. That count used to be computed
by a correlated subquery per candidate row, which made the query's cost grow with
``len(source_memory_ids)`` — unbounded, because consolidation appends to that
array and never prunes it (#1725). It is now computed set-wise. These tests pin
the *semantics* that rewrite has to preserve: the score is the number of distinct
shared source facts, and it orders the results.
"""

import uuid
from datetime import datetime, timezone

import pytest


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


@pytest.mark.asyncio
async def test_score_counts_distinct_shared_sources(memory, request_context):
    """Score == number of distinct source facts shared with the seed neighbourhood."""
    from hindsight_api.engine.db.ops import UpdatedWindow
    from hindsight_api.engine.task_backend import fq_table

    bank_id = f"test_obs_score_{uuid.uuid4().hex[:8]}"
    try:
        pool = await memory._get_pool()
        backend = await memory._get_backend()
        mu, ue, ml = fq_table("memory_units"), fq_table("unit_entities"), fq_table("memory_links")

        async with pool.acquire() as conn:
            await _ensure_bank(conn, bank_id)
            # Four world facts, all sharing one entity so they land in the
            # seed's connected-source set.
            facts = [await _insert_unit(conn, mu, bank_id, f"fact {i}", "world") for i in range(4)]
            entity_id = uuid.uuid4()
            await conn.execute(
                f"INSERT INTO {fq_table('entities')} (id, bank_id, canonical_name) VALUES ($1, $2, $3)",
                entity_id,
                bank_id,
                "Acme",
            )
            for fid in facts:
                await conn.execute(
                    f"INSERT INTO {ue} (unit_id, entity_id) VALUES ($1, $2)",
                    fid,
                    entity_id,
                )

            # The seed observation is built from fact 0 only; the candidates
            # share a varying number of the *other* facts.
            seed = await _insert_unit(conn, mu, bank_id, "seed obs", "observation", [facts[0]])
            three = await _insert_unit(conn, mu, bank_id, "shares three", "observation", facts[1:4])
            one = await _insert_unit(conn, mu, bank_id, "shares one", "observation", [facts[1]])
            # Duplicate ids in the array must count once — the old query used
            # COUNT(DISTINCT ...) and the rewrite must not turn that into COUNT(*).
            dupes = await _insert_unit(
                conn, mu, bank_id, "shares one, listed twice", "observation", [facts[2], facts[2]]
            )

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

        scores = {r["id"]: r["score"] for r in rows.entity}
        assert scores[three] == 3.0
        assert scores[one] == 1.0
        assert scores[dupes] == 1.0, "repeated source ids must count once"
        assert seed not in scores, "the seed itself must not come back as a result"

        ranked = [r["id"] for r in rows.entity]
        assert ranked[0] == three, "higher shared-source count must rank first"
    finally:
        await memory.delete_bank(bank_id=bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_wide_source_arrays_do_not_change_results(memory, request_context):
    """A long source array is a cost problem, not a correctness one (#3085).

    Two observations sharing the same single source fact must score the same
    whether their arrays hold 1 id or 200 — the extra ids are simply unrelated.
    """
    from hindsight_api.engine.db.ops import UpdatedWindow
    from hindsight_api.engine.task_backend import fq_table

    bank_id = f"test_obs_wide_{uuid.uuid4().hex[:8]}"
    try:
        pool = await memory._get_pool()
        backend = await memory._get_backend()
        mu, ue, ml = fq_table("memory_units"), fq_table("unit_entities"), fq_table("memory_links")

        async with pool.acquire() as conn:
            await _ensure_bank(conn, bank_id)
            shared = [await _insert_unit(conn, mu, bank_id, f"shared fact {i}", "world") for i in range(2)]
            entity_id = uuid.uuid4()
            await conn.execute(
                f"INSERT INTO {fq_table('entities')} (id, bank_id, canonical_name) VALUES ($1, $2, $3)",
                entity_id,
                bank_id,
                "Acme",
            )
            for fid in shared:
                await conn.execute(f"INSERT INTO {ue} (unit_id, entity_id) VALUES ($1, $2)", fid, entity_id)

            # Unrelated facts: no entity rows, so they never enter connected_sources.
            padding = [await _insert_unit(conn, mu, bank_id, f"unrelated {i}", "world") for i in range(200)]

            seed = await _insert_unit(conn, mu, bank_id, "seed obs", "observation", [shared[0]])
            narrow = await _insert_unit(conn, mu, bank_id, "narrow", "observation", [shared[1]])
            wide = await _insert_unit(conn, mu, bank_id, "wide", "observation", [shared[1], *padding])

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

        scores = {r["id"]: r["score"] for r in rows.entity}
        assert scores[narrow] == scores[wide] == 1.0
    finally:
        await memory.delete_bank(bank_id=bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_per_entity_cap_bounds_hub_traversal(memory, request_context):
    """A hub entity contributes at most ``per_entity_limit`` source facts (#3510).

    The cap used to be a LATERAL + LIMIT and is now a row_number() window, because
    the planner cannot estimate DISTINCT over a LIMIT subquery and mis-planned the
    scoring join into a nested loop. The two forms have to select the *same* rows:
    the highest ``per_entity_limit`` unit_ids of each entity. Ranking is by unit_id
    descending, which is what the LATERAL ordered by, so a candidate built from the
    lowest ids of an over-cap entity must fall outside the cap and score nothing.
    """
    from hindsight_api.engine.db.ops import UpdatedWindow
    from hindsight_api.engine.task_backend import fq_table

    bank_id = f"test_obs_cap_{uuid.uuid4().hex[:8]}"
    per_entity_limit = 3
    try:
        pool = await memory._get_pool()
        backend = await memory._get_backend()
        mu, ue, ml = fq_table("memory_units"), fq_table("unit_entities"), fq_table("memory_links")

        async with pool.acquire() as conn:
            await _ensure_bank(conn, bank_id)
            entity_id = uuid.uuid4()
            await conn.execute(
                f"INSERT INTO {fq_table('entities')} (id, bank_id, canonical_name) VALUES ($1, $2, $3)",
                entity_id,
                bank_id,
                "Hub",
            )
            # Six facts on one entity with ids we control, so "top 3 by unit_id
            # descending" is a known set rather than an accident of uuid4().
            facts = sorted(uuid.UUID(int=i) for i in range(1, 7))
            for fid in facts:
                await conn.execute(
                    f"""
                    INSERT INTO {mu} (id, bank_id, text, fact_type, source_memory_ids, event_date)
                    VALUES ($1, $2, $3, 'world', NULL, $4)
                    """,
                    fid,
                    bank_id,
                    f"hub fact {fid}",
                    datetime.now(timezone.utc),
                )
                await conn.execute(f"INSERT INTO {ue} (unit_id, entity_id) VALUES ($1, $2)", fid, entity_id)

            # The seed reaches the hub through its own source fact.
            seed = await _insert_unit(conn, mu, bank_id, "seed obs", "observation", [facts[0]])
            # Inside the cap: built from the highest ids. Outside: the lowest.
            inside = await _insert_unit(conn, mu, bank_id, "inside cap", "observation", facts[-3:])
            outside = await _insert_unit(conn, mu, bank_id, "outside cap", "observation", [facts[1]])

            rows = await backend.ops.expand_observations(
                conn,
                mu,
                ue,
                ml,
                [seed],
                100,
                per_entity_limit,
                UpdatedWindow(after=None, before=None, first_param_index=3),
            )

        scores = {r["id"] for r in rows.entity}
        assert inside in scores, "observations built from the top-ranked ids must be reachable"
        assert outside not in scores, "the per-entity cap must exclude ids ranked below it"
    finally:
        await memory.delete_bank(bank_id=bank_id, request_context=request_context)
