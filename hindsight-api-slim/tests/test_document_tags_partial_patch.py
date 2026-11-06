"""A document's tags can be changed one at a time with PATCH (#6339).

``tags`` replaces the whole set; ``add_tags`` / ``remove_tags`` change only the listed tags and
keep the rest — on the document and on every memory extracted from it.
"""

import uuid

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.memories import get_memories
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.schema import fq_table


async def _fact_tags(memory: MemoryEngine, bank_id: str) -> list[set[str]]:
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        page = await get_memories().scan_memories(
            conn=conn, fq_table=fq_table, bank_id=bank_id, fact_types=["experience", "world"], limit=1_000_000
        )
    return [set(f.tags or []) for f in page.memories]


@pytest.mark.asyncio
async def test_http_patch_adds_and_removes_single_tags(
    api_client, memory: MemoryEngine, request_context: RequestContext
):
    bank_id = f"test_patch_tags_partial_{uuid.uuid4().hex[:8]}"
    try:
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {
                    "content": "Alice said she prefers dark roast coffee and drinks it every morning.",
                    "document_id": "doc",
                    "tags": ["team-a", "team-b"],
                }
            ],
            request_context=request_context,
        )
        url = f"/v1/default/banks/{bank_id}/documents/doc"

        response = await api_client.patch(url, json={"add_tags": ["team-c"]})
        assert response.status_code == 200, response.text
        assert set((await api_client.get(url)).json()["tags"]) == {"team-a", "team-b", "team-c"}

        response = await api_client.patch(url, json={"remove_tags": ["team-a"], "add_tags": ["team-d"]})
        assert response.status_code == 200, response.text
        assert set((await api_client.get(url)).json()["tags"]) == {"team-b", "team-c", "team-d"}

        facts = await _fact_tags(memory, bank_id)
        assert facts, "Setup: retain should have produced facts"
        assert all(tags == {"team-b", "team-c", "team-d"} for tags in facts), facts

        both = await api_client.patch(url, json={"tags": ["x"], "add_tags": ["y"]})
        assert both.status_code == 422, both.text
        assert set((await api_client.get(url)).json()["tags"]) == {"team-b", "team-c", "team-d"}

        missing = await api_client.patch(f"/v1/default/banks/{bank_id}/documents/nope", json={"add_tags": ["x"]})
        assert missing.status_code == 404
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
