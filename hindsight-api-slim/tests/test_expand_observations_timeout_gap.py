"""Regression test: `_expand_observations()`'s entity-expansion query now
degrades gracefully on timeout, matching its sibling `_expand_combined()`.

Background
----------
PR #911 ("fix(recall): cap entity fanout in graph expansion") added two
protections to `LinkExpansionRetriever._expand_combined()` in
`hindsight_api/engine/search/link_expansion_retrieval.py`:

1. A LATERAL per-entity cap (`link_expansion_per_entity_limit`, default 200).
2. An `asyncio.wait_for(..., timeout=config.link_expansion_timeout)` wrapper
   (default 10s), with `except asyncio.TimeoutError:` falling back to
   semantic+causal-only results instead of failing the whole recall.

`_expand_observations()` (the sibling path for `fact_type="observation"`)
received protection #1 but not #2 — its `ops.expand_observations()` call was
never wrapped in a timeout/fallback, so a slow run on a large/dense bank
(very high entity fanout even under the per-entity cap — see the "issue
#3510" notes in `ops_postgresql.py::expand_observations`) propagated a raw
`TimeoutError` all the way up and failed the entire recall call:
`RuntimeError: Failed to search memories (TimeoutError): TimeoutError()`,
observed in production on a bank with ~410K `entity_cooccurrences` rows.

This test seeds directly via SQL (no LLM, no waiting on real consolidation)
and forces the real entity-expansion query inside `expand_observations()` to
exceed `link_expansion_timeout` — the same technique already established in
`test_graph_entity_fanout_cap.py::test_entity_expansion_timeout_fallback` for
the sibling path — and asserts recall now degrades instead of raising.
"""

import logging
import uuid

import pytest

from hindsight_api.engine.retain import embedding_utils


def _to_str(emb: list[float]) -> str:
    return "[" + ",".join(str(v) for v in emb) + "]"


async def _seed_world_fact_and_observation(memory, bank_id: str) -> dict:
    """Seeds directly via SQL -- same technique as
    `test_recall_pg_enrichment_combo.py::seeded_combo`: a `world` fact with an
    entity link, plus an `observation` unit consolidated from it via
    `source_memory_ids`, so a `fact_type=["observation"]` recall has a real
    semantic seed to expand from."""
    fact_id = str(uuid.uuid4())
    obs_id = str(uuid.uuid4())
    fact_text = "Alice migrated the billing service to the new cluster"
    obs_text = "Alice handles infrastructure migrations"

    embeddings = await embedding_utils.generate_embeddings_batch(memory.embeddings, [fact_text, obs_text])

    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        ent_id = await conn.fetchval(
            "INSERT INTO entities (bank_id, canonical_name, mention_count) VALUES ($1, $2, 1) RETURNING id",
            bank_id,
            "billing service",
        )
        await conn.execute(
            "INSERT INTO memory_units (id, bank_id, text, fact_type, embedding, event_date) "
            "VALUES ($1, $2, $3, 'world', $4::vector, now())",
            fact_id,
            bank_id,
            fact_text,
            _to_str(embeddings[0]),
        )
        await conn.execute(
            "INSERT INTO unit_entities (unit_id, entity_id) VALUES ($1, $2)",
            fact_id,
            ent_id,
        )
        await conn.execute(
            "INSERT INTO memory_units (id, bank_id, text, fact_type, embedding, event_date, "
            "source_memory_ids, proof_count) VALUES ($1, $2, $3, 'observation', $4::vector, now(), $5::uuid[], 1)",
            obs_id,
            bank_id,
            obs_text,
            _to_str(embeddings[1]),
            [fact_id],
        )

    return {"fact_id": fact_id, "obs_id": obs_id}


@pytest.mark.asyncio
async def test_expand_observations_falls_back_gracefully_on_timeout(memory, request_context, caplog):
    """FIX: forcing link_expansion_timeout low enough that the real entity
    CTE inside expand_observations() is guaranteed to exceed it should
    degrade to semantic+causal results, not fail the whole recall call.

    The recall-succeeds assertion alone doesn't discriminate this fix from
    its absence: the seed observation is already found via direct semantic
    search regardless of whether entity expansion ran, times out, or is
    never guarded at all. The caplog assertion below is what actually pins
    the fix -- the warning is only emitted from the `except
    asyncio.TimeoutError` branch this PR adds around the entity CTE, so it
    can only fire when that branch exists and was taken.
    """
    bank_id = f"test-expand-obs-timeout-fallback-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id, request_context=request_context)

    try:
        await _seed_world_fact_and_observation(memory, bank_id)

        from hindsight_api.config import _get_raw_config
        from hindsight_api.engine.memory_engine import Budget

        config = _get_raw_config()
        original_timeout = config.link_expansion_timeout
        try:
            config.link_expansion_timeout = 0.0001  # guaranteed to be exceeded

            with caplog.at_level(logging.WARNING, logger="hindsight_api.engine.db.ops_postgresql"):
                result = await memory.recall_async(
                    bank_id=bank_id,
                    query="Alice billing service migration",
                    fact_type=["observation"],
                    budget=Budget.MID,
                    max_tokens=2048,
                    request_context=request_context,
                    _quiet=True,
                )

            # Recall succeeds even though entity expansion timed out; the
            # seed observation itself is still found via semantic search.
            assert result.results is not None
            assert len(result.results) > 0

            # Pin the actual fix: the timeout/fallback guard must have fired.
            assert any("[ExpandObservations] Entity expansion timed out" in r.message for r in caplog.records), (
                f"expected the entity-expansion timeout warning, got: {[r.message for r in caplog.records]}"
            )
        finally:
            config.link_expansion_timeout = original_timeout
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_expand_combined_gracefully_falls_back_on_timeout(memory, request_context):
    """CONTRAST / sanity check: the non-observation path's existing guard
    (added by PR #911) still behaves the same way after this change."""
    bank_id = f"test-expand-combined-timeout-ok-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id, request_context=request_context)

    try:
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {
                    "content": "Alice works on the backend API at TechCorp",
                    "context": "team info",
                    "entities": [{"text": "Alice"}, {"text": "TechCorp"}],
                },
            ],
            request_context=request_context,
        )

        from hindsight_api.config import _get_raw_config
        from hindsight_api.engine.memory_engine import Budget

        config = _get_raw_config()
        original_timeout = config.link_expansion_timeout
        try:
            config.link_expansion_timeout = 0.0001  # guaranteed to be exceeded

            result = await memory.recall_async(
                bank_id=bank_id,
                query="Alice",
                fact_type=["world"],
                budget=Budget.MID,
                max_tokens=2048,
                request_context=request_context,
                _quiet=True,
            )
            assert result.results is not None
            assert len(result.results) > 0
        finally:
            config.link_expansion_timeout = original_timeout
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
