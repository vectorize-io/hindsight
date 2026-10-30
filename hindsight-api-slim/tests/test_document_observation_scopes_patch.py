"""Moving an existing document to another observation scoping with PATCH (#5309).

The reporter's bank retained every session under a ``session:<id>`` tag with the default
``combined`` scoping, so each session consolidated alone. New retains can send ``shared``;
these tests pin that the old documents can follow without re-extraction:

- the facts move, their per-session observations go, and the next consolidation rebuilds
  them in the new scope — keeping the facts' own tags;
- only memories whose RESOLVED scopes change are requeued (the #5593 rule): a different
  spelling of the same scopes, or re-sending the current value, rebuilds nothing;
- the new scoping is written into the document's retain params, so reprocess keeps it.
"""

import asyncio
import uuid

import pytest

from hindsight_api import RequestContext
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.memories import FactRecord, get_memories
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.schema import fq_table

_SESSION_A = "Alice said she prefers dark roast coffee and drinks it every morning before work."
_SESSION_B = "Alice mentioned she switched to dark roast coffee because light roast upsets her stomach."


@pytest.fixture(autouse=True)
def enable_observations():
    config = _get_raw_config()
    original = config.enable_observations
    config.enable_observations = True
    yield
    config.enable_observations = original


async def _scan(memory: MemoryEngine, bank_id: str, fact_types: list[str]) -> list[FactRecord]:
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        page = await get_memories().scan_memories(
            conn=conn, fq_table=fq_table, bank_id=bank_id, fact_types=fact_types, limit=1_000_000
        )
    return list(page.memories)


async def _retain_session(memory: MemoryEngine, bank_id: str, session: str, content: str, ctx: RequestContext) -> None:
    # Auto-consolidation runs inline under the tests' synchronous task backend; it would mark the
    # requeued facts consolidated again before a test could see that they were requeued.
    await memory.update_bank_config(bank_id, {"enable_auto_consolidation": False}, request_context=ctx)
    await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[{"content": content, "document_id": f"session-{session}", "tags": [f"session:{session}"]}],
        request_context=ctx,
    )


@pytest.mark.asyncio
async def test_patch_moves_per_session_documents_to_shared(memory: MemoryEngine, request_context: RequestContext):
    bank_id = f"test_patch_scopes_{uuid.uuid4().hex[:8]}"
    try:
        await _retain_session(memory, bank_id, "a", _SESSION_A, request_context)
        await _retain_session(memory, bank_id, "b", _SESSION_B, request_context)
        await memory.run_consolidation(bank_id=bank_id, request_context=request_context)
        before = await _scan(memory, bank_id, ["observation"])
        assert before and all(o.tags for o in before), "Setup: per-session observations, each tagged"

        for session in ("a", "b"):
            assert await memory.update_document(
                f"session-{session}", bank_id, observation_scopes="shared", request_context=request_context
            )

        facts = await _scan(memory, bank_id, ["experience", "world"])
        assert facts and all(f.observation_scopes == "shared" for f in facts)
        assert {tuple(f.tags or []) for f in facts} == {("session:a",), ("session:b",)}, "Facts keep their tags"
        assert all(f.consolidated_at is None for f in facts), "Every moved fact is queued again"
        remaining = {o.unit_id for o in await _scan(memory, bank_id, ["observation"])}
        assert not remaining & {o.unit_id for o in before}, "The per-session observations are gone"

        doc = await memory.get_document("session-a", bank_id, request_context=request_context)
        assert doc is not None and doc["observation_scopes"] == "shared"
        assert doc["tags"] == ["session:a"], "A scope-only PATCH leaves the document's tags alone"

        await memory.run_consolidation(bank_id=bank_id, request_context=request_context)
        after = await _scan(memory, bank_id, ["observation"])
        assert after and all(not o.tags for o in after), "Rebuilt observations live in the shared scope"

        # Re-sending the value the documents already have moves nothing.
        assert await memory.update_document(
            "session-a", bank_id, observation_scopes="shared", request_context=request_context
        )
        assert {o.unit_id for o in await _scan(memory, bank_id, ["observation"])} == {o.unit_id for o in after}
        assert all(f.consolidated_at is not None for f in await _scan(memory, bank_id, ["experience", "world"]))
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_patch_to_the_same_resolved_scopes_requeues_nothing(
    memory: MemoryEngine, request_context: RequestContext
):
    """``combined`` over one tag and ``[[that tag]]`` write to the same scope: the stored spec
    changes, but no observation is deleted and no fact is queued again."""
    bank_id = f"test_patch_scopes_same_{uuid.uuid4().hex[:8]}"
    try:
        await _retain_session(memory, bank_id, "a", _SESSION_A, request_context)
        await memory.run_consolidation(bank_id=bank_id, request_context=request_context)
        before = {o.unit_id for o in await _scan(memory, bank_id, ["observation"])}
        assert before, "Setup: consolidation should have produced observations"

        assert await memory.update_document(
            "session-a", bank_id, observation_scopes=[["session:a"]], request_context=request_context
        )

        facts = await _scan(memory, bank_id, ["experience", "world"])
        assert facts and all(f.observation_scopes == [["session:a"]] for f in facts), "The new spec is stored"
        assert all(f.consolidated_at is not None for f in facts), "Nothing moved, so nothing is requeued"
        assert {o.unit_id for o in await _scan(memory, bank_id, ["observation"])} == before
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_reprocess_keeps_the_patched_scopes(memory: MemoryEngine, request_context: RequestContext):
    bank_id = f"test_patch_scopes_reprocess_{uuid.uuid4().hex[:8]}"
    try:
        await _retain_session(memory, bank_id, "a", _SESSION_A, request_context)
        assert await memory.update_document(
            "session-a", bank_id, observation_scopes="shared", request_context=request_context
        )
        before = {f.unit_id for f in await _scan(memory, bank_id, ["experience", "world"])}

        await memory.reprocess_document(bank_id=bank_id, document_id="session-a", request_context=request_context)
        await asyncio.sleep(0.5)

        facts = await _scan(memory, bank_id, ["experience", "world"])
        assert facts and not ({f.unit_id for f in facts} & before), "Setup: reprocess re-extracted the facts"
        assert all(f.observation_scopes == "shared" for f in facts), "Reprocess brought the old scoping back"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_http_patch_accepts_observation_scopes(api_client, memory: MemoryEngine, request_context):
    bank_id = f"test_patch_scopes_http_{uuid.uuid4().hex[:8]}"
    try:
        await _retain_session(memory, bank_id, "a", _SESSION_A, request_context)
        url = f"/v1/default/banks/{bank_id}/documents/session-a"

        response = await api_client.patch(url, json={"observation_scopes": "shared"})
        assert response.status_code == 200, response.text
        assert (await api_client.get(url)).json()["observation_scopes"] == "shared"

        assert (await api_client.patch(url, json={"observation_scopes": "everything"})).status_code == 422
        assert (await api_client.patch(url, json={})).status_code == 422
        missing = await api_client.patch(
            f"/v1/default/banks/{bank_id}/documents/nope", json={"observation_scopes": "shared"}
        )
        assert missing.status_code == 404
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
