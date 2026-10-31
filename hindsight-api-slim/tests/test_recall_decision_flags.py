"""Recall's prune and dedup reach the reranker: the request wins, then the bank, then the server."""

import asyncio
import uuid

import numpy as np
import pytest

from hindsight_api import MemoryEngine, RequestContext
from hindsight_api.engine.cross_encoder import CrossEncoderModel
from hindsight_api.engine.search.reranking import CrossEncoderReranker, RerankResult
from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult

RC = RequestContext(tenant_id="default")


@pytest.mark.memory_backend_incompatible
async def test_request_overrides_bank_overrides_server(memory_no_llm_verify: MemoryEngine, monkeypatch):
    engine = memory_no_llm_verify
    bank_id = f"test-decision-flags-{uuid.uuid4().hex[:8]}"
    await engine.ensure_bank_profile(bank_id, request_context=RC)
    seen: list[tuple[bool, bool]] = []

    async def spy(query, candidates, now=None, prune=False, dedup=False, embeddings=None):
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
async def test_dedup_reads_the_candidates_embeddings(memory_no_llm_verify: MemoryEngine, monkeypatch):
    """This bank's stored embeddings of the ids asked for; a memory without one is left out, and
    a failed read falls back to None, which makes dedup ask every pair instead of failing."""
    engine = memory_no_llm_verify
    bank_id, other_bank = f"test-dedup-emb-{uuid.uuid4().hex[:8]}", f"test-dedup-other-{uuid.uuid4().hex[:8]}"
    a, b, bare, elsewhere = (str(uuid.uuid4()) for _ in range(4))
    vector = (await engine.embeddings.encode(["tea"]))[0]
    as_text = "[" + ",".join(str(x) for x in vector) + "]"
    pool = await engine._get_pool()
    # Direct insert: the API can't store a memory without an embedding, and this checks the
    # read stays in its own bank.
    async with pool.acquire() as conn:
        for bank in (bank_id, other_bank):
            await engine.ensure_bank_profile(bank, request_context=RC)
        for unit_id, bank, embedding in (
            (a, bank_id, as_text),
            (b, bank_id, as_text),
            (bare, bank_id, None),
            (elsewhere, other_bank, as_text),
        ):
            await conn.execute(
                "INSERT INTO memory_units (id, bank_id, text, fact_type, embedding) VALUES ($1, $2, $3, 'world', $4::vector)",
                unit_id,
                bank,
                unit_id,
                embedding,
            )

    found = await engine._candidate_embeddings(bank_id, [a, b, bare, elsewhere])
    assert found is not None and set(found) == {a, b}
    assert np.allclose(found[a], vector, atol=1e-6)

    from hindsight_api.engine import memories

    async def broken(**kwargs):
        raise RuntimeError("no embeddings here")

    monkeypatch.setattr(memories.get_memories(), "unit_embeddings", broken)
    assert await engine._candidate_embeddings(bank_id, [a, b]) is None


class _CrossEncoder(CrossEncoderModel):
    """A plain cross-encoder: it only orders, and scores the last candidate 0.0."""

    @property
    def provider_name(self) -> str:
        return "local"

    async def initialize(self) -> None:
        pass

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [(len(pairs) - 1 - i) / len(pairs) for i in range(len(pairs))]


def _candidate(unit_id: str, text: str) -> MergedCandidate:
    return MergedCandidate(
        retrieval=RetrievalResult(id=unit_id, text=text, fact_type="world"), rrf_score=0.0, source_ranks={}
    )


async def test_a_reranker_without_prune_or_dedup_ignores_them():
    """Asked of a cross-encoder, prune and dedup change nothing: a candidate scored 0.0 and a
    repeated text both stay, in the same order, and the embeddings read is cancelled unawaited."""
    reranker = CrossEncoderReranker(cross_encoder=_CrossEncoder())
    candidates = [_candidate("a", "Marta likes sencha"), _candidate("b", "Marta likes sencha"), _candidate("c", "x")]
    plain = await reranker.rerank("q", candidates)

    async def never() -> None:
        raise AssertionError("dedup read the embeddings for a cross-encoder")

    embeddings = asyncio.create_task(never())
    flagged = await reranker.rerank("q", candidates, prune=True, dedup=True, embeddings=embeddings)

    assert [(r.id, r.weight) for r in flagged.results] == [(r.id, r.weight) for r in plain.results]
    assert [r.id for r in flagged.results] == ["a", "b", "c"]
    assert flagged.results[-1].weight == 0.0
    with pytest.raises(asyncio.CancelledError):
        await embeddings


@pytest.mark.memory_backend_incompatible
async def test_recall_with_prune_and_dedup_on_a_cross_encoder_returns_the_same(
    memory_no_llm_verify: MemoryEngine, monkeypatch
):
    """End to end through recall: same results as without the flags, and no embeddings read."""
    engine = memory_no_llm_verify
    bank_id = f"test-flags-ignored-{uuid.uuid4().hex[:8]}"
    await engine.ensure_bank_profile(bank_id, request_context=RC)
    await engine.retain_batch_async(
        bank_id=bank_id,
        contents=[{"content": "Marta likes sencha."}, {"content": "Marta likes sencha."}, {"content": "Leo runs."}],
        request_context=RC,
    )
    monkeypatch.setattr(engine._cross_encoder_reranker, "cross_encoder", _CrossEncoder())

    async def no_read(*args, **kwargs):
        raise AssertionError("dedup read the embeddings for a cross-encoder")

    monkeypatch.setattr(engine, "_candidate_embeddings", no_read)
    plain = await engine.recall_async(bank_id=bank_id, query="tea", request_context=RC)
    flagged = await engine.recall_async(bank_id=bank_id, query="tea", request_context=RC, prune=True, dedup=True)
    assert plain.results
    assert [r.id for r in flagged.results] == [r.id for r in plain.results]
