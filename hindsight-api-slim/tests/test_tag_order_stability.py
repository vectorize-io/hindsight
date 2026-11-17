"""#6591: tag order must never read as a change.

String hashing differs per process, so tags that went through a ``set`` came out in a
different order on each worker. Every place that compared tag lists in order then saw a
change where there was none — an append rewrote every unit of its document — and the
mental-model ``exact`` filter missed a model whose tags were stored in another order.
"""

import uuid

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.retain.orchestrator import _build_retain_params
from tests.test_memories_extension import InMemoryMemories, StoredMemory


_TAGS = ["zeta", "alpha", "mid"]
_CONTENT = " ".join(f"The Grandview property offers mobile room key access on floor {i}." for i in range(20))


def test_merged_tags_come_out_in_one_order():
    _params, merged = _build_retain_params([{"content": "x", "tags": ["zeta", "mid"]}], ["alpha", "zeta"])
    assert merged == ["alpha", "mid", "zeta"]


@pytest.mark.asyncio
async def test_relabel_with_reordered_tags_patches_nothing():
    store = InMemoryMemories({})
    store.rows["fact-1"] = StoredMemory(
        unit_id="fact-1", text="a fact", fact_type="world", document_id="doc-1", tags=list(reversed(_TAGS))
    )
    result = await store.relabel_document_memories(
        conn=None,
        fq_table=None,
        bank_id="bank-x",
        document_id="doc-1",
        tags=_TAGS,
        metadata={},
        observation_scopes=None,
        final_tags=lambda _old: list(_TAGS),
    )
    assert result.updated == 0


@pytest.mark.asyncio
# Stages the case by rewriting the `memory_units` column directly, which an alternative store
# never reads (see test_memory_units_updated_at.py).
@pytest.mark.memory_backend_incompatible
async def test_re_retain_does_not_rewrite_units_stored_in_another_tag_order(
    memory: MemoryEngine, request_context: RequestContext
):
    """The reporter's case: another worker wrote the document's tags in another order."""
    bank_id = f"test_tag_order_{uuid.uuid4().hex[:8]}"
    item = {"content": _CONTENT, "context": "hotel", "document_id": "doc-1", "tags": list(_TAGS)}
    try:
        await memory.retain_batch_async(
            bank_id=bank_id, contents=[item], document_tags=list(_TAGS), request_context=request_context
        )
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            # What a worker with a different hash seed leaves behind.
            await conn.execute(
                "UPDATE memory_units SET tags = $2 WHERE bank_id = $1",
                bank_id,
                list(reversed(sorted(_TAGS))),
            )
            before = await conn.fetch("SELECT id, updated_at FROM memory_units WHERE bank_id = $1", bank_id)
        assert before, "Setup: retain should have stored units"

        await memory.retain_batch_async(
            bank_id=bank_id, contents=[item], document_tags=list(_TAGS), request_context=request_context
        )

        async with pool.acquire() as conn:
            after = await conn.fetch("SELECT id, updated_at FROM memory_units WHERE bank_id = $1", bank_id)
        assert {r["id"]: r["updated_at"] for r in after} == {r["id"]: r["updated_at"] for r in before}
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_exact_mental_model_filter_ignores_tag_order(memory: MemoryEngine, request_context: RequestContext):
    bank_id = f"test_tag_order_mm_{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        await memory.create_mental_model(
            bank_id=bank_id,
            name="Model",
            source_query="Query",
            content="Content",
            tags=list(_TAGS),
            request_context=request_context,
        )
        page = await memory.list_mental_models(
            bank_id=bank_id, tags=list(reversed(_TAGS)), tags_match="exact", request_context=request_context
        )
        assert page.total == 1
        page = await memory.list_mental_models(
            bank_id=bank_id, tags=_TAGS[:2], tags_match="exact", request_context=request_context
        )
        assert page.total == 0
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
