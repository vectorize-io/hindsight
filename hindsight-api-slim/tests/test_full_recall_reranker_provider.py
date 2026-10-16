"""``FullRecallRequest.reranker_provider`` names the engine's reranker.

A store that answers a whole recall reranks with a cross-encoder of its own. It can only stand in
for a pointwise cross-encoder: a decision model or a rank-position reranker judges the pool as a
whole, so a store has to see which one the engine would call in order to decline it.
"""

import uuid
from collections.abc import Iterator

import pytest

from hindsight_api.engine.memories import set_memories
from hindsight_api.engine.memories.base import FullRecallRequest
from hindsight_api.engine.response_models import RecallResult
from tests.test_memories_extension import InMemoryMemories


class _RecordingStore(InMemoryMemories):
    def __init__(self) -> None:
        super().__init__({})
        self.seen: object = "not-called"

    async def full_recall(self, request: FullRecallRequest) -> RecallResult:
        self.seen = request.reranker_provider
        return RecallResult(results=[])


@pytest.fixture
def restore_default_store() -> Iterator[None]:
    yield
    set_memories(None)


@pytest.mark.asyncio
async def test_the_store_is_told_which_reranker_the_engine_would_call(
    api_client, memory, request_context, restore_default_store
):
    bank_id = f"so-reranker-{uuid.uuid4().hex[:8]}"
    store = _RecordingStore()
    set_memories(store)
    await memory.ensure_bank_profile(bank_id, request_context=request_context)

    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories/recall",
        json={"query": "what shipped", "types": ["world"], "limit": 10},
    )

    assert response.status_code == 200, response.text
    expected = memory._cross_encoder_reranker.cross_encoder.primary_provider_name
    assert expected
    assert store.seen == expected
