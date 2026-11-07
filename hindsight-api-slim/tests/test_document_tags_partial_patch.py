"""A document's tags can be changed one at a time with PATCH (#6339).

``tags`` replaces the whole set; ``add_tags`` / ``remove_tags`` change only the listed tags and
keep the rest — on the document and on every memory extracted from it.
"""

import asyncio
import uuid

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.memories import get_memories
from hindsight_api.engine.memories.base import DocumentTags
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.extensions import OperationValidationError


async def _retain(memory: MemoryEngine, bank_id: str, tags: list[str], ctx: RequestContext) -> None:
    await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[
            {
                "content": "Alice said she prefers dark roast coffee and drinks it every morning.",
                "document_id": "doc",
                "tags": tags,
            }
        ],
        request_context=ctx,
    )


async def _fact_tags(memory: MemoryEngine, bank_id: str, ctx: RequestContext) -> list[set[str]]:
    units = await memory.list_memory_units(
        bank_id, fact_type=["world", "experience"], document_id="doc", request_context=ctx
    )
    return [set(u["tags"] or []) for u in units["items"]]


@pytest.mark.asyncio
async def test_http_patch_adds_and_removes_single_tags(
    api_client, memory: MemoryEngine, request_context: RequestContext
):
    bank_id = f"test_patch_tags_partial_{uuid.uuid4().hex[:8]}"
    try:
        await _retain(memory, bank_id, ["team-a", "team-b"], request_context)
        url = f"/v1/default/banks/{bank_id}/documents/doc"

        async def tags() -> set[str]:
            return set((await api_client.get(url)).json()["tags"])

        response = await api_client.patch(url, json={"add_tags": ["team-c"]})
        assert response.status_code == 200, response.text
        assert await tags() == {"team-a", "team-b", "team-c"}

        response = await api_client.patch(url, json={"remove_tags": ["team-a"], "add_tags": ["team-d"]})
        assert response.status_code == 200, response.text
        assert await tags() == {"team-b", "team-c", "team-d"}

        facts = await _fact_tags(memory, bank_id, request_context)
        assert facts, "Setup: retain should have produced facts"
        assert all(t == {"team-b", "team-c", "team-d"} for t in facts), facts

        # Removing a tag the document lacks changes nothing; a tag both removed and added stays.
        assert (await api_client.patch(url, json={"remove_tags": ["nope"]})).status_code == 200
        assert (
            await api_client.patch(url, json={"remove_tags": ["team-b"], "add_tags": ["team-b"]})
        ).status_code == 200
        assert await tags() == {"team-b", "team-c", "team-d"}

        both = await api_client.patch(url, json={"tags": ["x"], "add_tags": ["y"]})
        assert both.status_code == 422, both.text
        assert await tags() == {"team-b", "team-c", "team-d"}

        # Removing every tag leaves the document (and its memories) untagged.
        response = await api_client.patch(url, json={"remove_tags": ["team-b", "team-c", "team-d"]})
        assert response.status_code == 200, response.text
        assert await tags() == set()
        assert all(t == set() for t in await _fact_tags(memory, bank_id, request_context))

        missing = await api_client.patch(f"/v1/default/banks/{bank_id}/documents/nope", json={"add_tags": ["x"]})
        assert missing.status_code == 404
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_concurrent_partial_changes_keep_each_others_tags(memory: MemoryEngine, request_context: RequestContext):
    """Each partial change is computed from the tags it reads; two at once must not drop the
    other's tag by writing a set read before it landed."""
    bank_id = f"test_patch_tags_concurrent_{uuid.uuid4().hex[:8]}"
    try:
        await _retain(memory, bank_id, ["base"], request_context)
        added = [f"t{i}" for i in range(6)]

        await asyncio.gather(
            *(memory.update_document("doc", bank_id, add_tags=[t], request_context=request_context) for t in added)
        )

        doc = await memory.get_document("doc", bank_id, request_context=request_context)
        assert doc is not None
        assert set(doc["tags"]) == {"base", *added}
        facts = await _fact_tags(memory, bank_id, request_context)
        assert facts and all(t == {"base", *added} for t in facts), facts
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_partial_change_is_refused_when_the_store_cannot_read_the_tags(
    memory: MemoryEngine, request_context: RequestContext, monkeypatch
):
    """A store whose document record carries no tags cannot resolve a partial change: refuse it
    rather than treat the unknown set as empty and wipe the document's tags."""
    bank_id = f"test_patch_tags_unreadable_{uuid.uuid4().hex[:8]}"
    try:
        await _retain(memory, bank_id, ["base"], request_context)

        async def tagless(**_kwargs) -> DocumentTags:
            return DocumentTags(found=True, tags=None)

        with monkeypatch.context() as patched:
            patched.setattr(get_memories(), "current_document_tags", tagless)
            with pytest.raises(OperationValidationError) as e:
                await memory.update_document("doc", bank_id, add_tags=["x"], request_context=request_context)
        assert e.value.status_code == 400

        doc = await memory.get_document("doc", bank_id, request_context=request_context)
        assert doc is not None and doc["tags"] == ["base"]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_concurrent_full_and_partial_changes_keep_memories_in_step(
    memory: MemoryEngine, request_context: RequestContext
):
    """A full `tags` write decides from the tags it reads whether the memories need retagging; read
    stale beside a partial change, it would leave the memories carrying a tag the document lost."""
    bank_id = f"test_patch_tags_mixed_{uuid.uuid4().hex[:8]}"
    try:
        await _retain(memory, bank_id, ["a"], request_context)

        for _ in range(3):
            await asyncio.gather(
                memory.update_document("doc", bank_id, tags=["a"], request_context=request_context),
                memory.update_document("doc", bank_id, add_tags=["b"], request_context=request_context),
            )
            doc = await memory.get_document("doc", bank_id, request_context=request_context)
            assert doc is not None
            facts = await _fact_tags(memory, bank_id, request_context)
            assert facts and all(t == set(doc["tags"]) for t in facts), (doc["tags"], facts)
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
