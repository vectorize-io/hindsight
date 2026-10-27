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

    async def spy(query, candidates, now=None, prune=False, dedup=False):
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
