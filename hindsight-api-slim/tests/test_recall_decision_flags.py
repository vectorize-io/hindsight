"""Recall's prune and dedup reach the reranker: the request wins, then the bank, then the server."""

import uuid

import pytest

from hindsight_api import MemoryEngine, RequestContext
from hindsight_api.engine.search.reranking import RerankResult

RC = RequestContext(tenant_id="default")


@pytest.mark.memory_backend_incompatible
async def test_request_overrides_bank_overrides_server(memory_no_llm_verify: MemoryEngine, monkeypatch):
    engine = memory_no_llm_verify
    bank_id = f"test-decision-flags-{uuid.uuid4().hex[:8]}"
    await engine.ensure_bank_profile(bank_id, request_context=RC)
    seen: list[tuple[bool, bool]] = []

    async def spy(query, candidates, now=None, prune=False, dedup=False, links=None):
        seen.append((prune, dedup))
        return RerankResult(results=[], provider_name="spy")

    monkeypatch.setattr(engine._cross_encoder_reranker, "rerank", spy)

    async def recall(**flags) -> tuple[bool, bool]:
        seen.clear()
        await engine.recall_async(bank_id=bank_id, query="q", request_context=RC, **flags)
        assert seen, "the reranker was not called"
        return seen[-1]

    # Unconfigured: the server default is off.
    assert await recall() == (False, False)

    await engine._config_resolver.update_bank_config(
        bank_id, {"recall_decision_model_prune": True, "recall_decision_model_dedup": True}
    )
    assert await recall() == (True, True)
    # An internal recall that passes nothing gets the bank's value too.
    assert await recall(prune=None, dedup=None) == (True, True)
    assert await recall(prune=False, dedup=False) == (False, False)
    assert await recall(prune=True, dedup=False) == (True, False)


@pytest.mark.memory_backend_incompatible
async def test_dedup_reads_the_semantic_links_among_the_candidates(memory_no_llm_verify: MemoryEngine, monkeypatch):
    """Only semantic links, both endpoints among the ids, this bank only; a failed read falls
    back to None, which makes dedup ask every pair instead of failing the recall."""
    engine = memory_no_llm_verify
    bank_id, other_bank = f"test-dedup-links-{uuid.uuid4().hex[:8]}", f"test-dedup-other-{uuid.uuid4().hex[:8]}"
    a, b, c, outside = (str(uuid.uuid4()) for _ in range(4))
    pool = await engine._get_pool()
    # Seeded directly: retain derives these links from real embeddings, which this test does
    # not need — it pins which stored rows the read returns.
    async with pool.acquire() as conn:
        for bank in (bank_id, other_bank):
            await engine.ensure_bank_profile(bank, request_context=RC)
        for unit_id, bank in ((a, bank_id), (b, bank_id), (c, bank_id), (outside, bank_id)):
            await conn.execute(
                "INSERT INTO memory_units (id, bank_id, text, fact_type) VALUES ($1, $2, $3, 'world')",
                unit_id,
                bank,
                unit_id,
            )
        for from_id, to_id, link_type, bank in (
            (a, b, "semantic", bank_id),
            (b, c, "temporal", bank_id),
            (c, outside, "semantic", bank_id),
            (c, a, "semantic", other_bank),
        ):
            await conn.execute(
                "INSERT INTO memory_links (from_unit_id, to_unit_id, link_type, weight, bank_id)"
                " VALUES ($1, $2, $3, 0.9, $4)",
                from_id,
                to_id,
                link_type,
                bank,
            )

    assert await engine._semantic_links_among(bank_id, [a, b, c]) == {frozenset((a, b))}

    from hindsight_api.engine import memories

    async def broken(**kwargs):
        raise RuntimeError("no links here")

    monkeypatch.setattr(memories.get_memories(), "semantic_links_among", broken)
    assert await engine._semantic_links_among(bank_id, [a, b, c]) is None
