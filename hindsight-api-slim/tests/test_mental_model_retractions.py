"""Retracted grounding: facts a document cites that no longer exist.

Split the way the project's testing note asks for. The mechanics here are
deterministic and asserted directly — which ids count as retracted, what gets
pruned, when the unsay pass runs and when it is deferred. What an LLM does with
the retraction prompt is not tested here: the pass is driven through a stub that
returns a fixed operation list, so these cover the pipeline, not the model.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.memories import get_memories
from hindsight_api.engine.memory_engine import MemoryEngine, fq_table
from hindsight_api.engine.reflect.retractions import (
    MEMORY_BACKED_FACT_TYPES,
    based_on_fact_ids,
    partition_retracted,
    prune_based_on,
)

# ---------------------------------------------------------------------------
# Pure helpers — no DB, no LLM
# ---------------------------------------------------------------------------


def _fact(fact_id: str, text: str = "a fact", fact_type: str = "observation") -> dict[str, Any]:
    return {"id": fact_id, "text": text, "type": fact_type, "context": None}


def test_based_on_fact_ids_collects_memory_backed_types_only():
    based_on = {
        "world": [_fact("w1")],
        "experience": [_fact("e1")],
        "observation": [_fact("o1")],
        "mental-models": [_fact("mm1")],
        "directives": [_fact("d1")],
    }
    assert based_on_fact_ids(based_on) == ["w1", "e1", "o1"]


def test_based_on_fact_ids_never_reports_non_memory_types():
    """The whole guard: mental-model and directive ids address other tables.

    Treating them as memory ids would report every one of them as missing and
    retract content grounded on a perfectly healthy sibling document.
    """
    assert "mental-models" not in MEMORY_BACKED_FACT_TYPES
    assert "directives" not in MEMORY_BACKED_FACT_TYPES
    assert based_on_fact_ids({"mental-models": [_fact("mm1")], "directives": [_fact("d1")]}) == []


def test_based_on_fact_ids_deduplicates_and_preserves_order():
    assert based_on_fact_ids({"world": [_fact("a"), _fact("b"), _fact("a")]}) == ["a", "b"]


@pytest.mark.parametrize(
    "based_on",
    [None, {}, {"world": None}, {"world": ["not-a-dict"]}, {"world": [{"no_id": 1}]}],
)
def test_based_on_fact_ids_tolerates_malformed_payloads(based_on):
    """A based_on from an older version, or hand-edited, must not break a refresh."""
    assert based_on_fact_ids(based_on) == []


def test_partition_retracted_reports_only_missing_memory_facts():
    based_on = {
        "observation": [_fact("live"), _fact("gone", "the retired claim")],
        "mental-models": [_fact("mm1")],
    }
    retracted = partition_retracted(based_on, live_ids={"live"})
    assert retracted.ids == {"gone"}
    # The text travels with it: the row is deleted, so this copy is all that is
    # left of what the document was told.
    assert retracted.facts[0]["text"] == "the retired claim"
    assert bool(retracted) is True


def test_partition_retracted_is_empty_when_everything_is_live():
    assert not partition_retracted({"observation": [_fact("a"), _fact("b")]}, live_ids={"a", "b"})


def test_prune_based_on_drops_named_ids_and_keeps_the_rest():
    based_on = {"observation": [_fact("keep"), _fact("drop")], "mental-models": [_fact("mm1")]}
    pruned = prune_based_on(based_on, {"drop"})
    assert [f["id"] for f in pruned["observation"]] == ["keep"]
    assert [f["id"] for f in pruned["mental-models"]] == ["mm1"]


# ---------------------------------------------------------------------------
# Fixtures shared by the DB-backed tests
# ---------------------------------------------------------------------------


async def _ensure_bank(memory: MemoryEngine, bank_id: str, request_context: RequestContext) -> None:
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)


async def _insert_memory(memory: MemoryEngine, conn, bank_id: str, text: str) -> uuid.UUID:
    """Seed one memory through the store, bypassing the LLM retain pipeline."""
    store = get_memories()
    fact = SimpleNamespace(
        fact_text=text,
        embedding=memory.embeddings.encode([text])[0],
        fact_type="world",
        tags=[],
        context=None,
        document_id=None,
        chunk_id=None,
        metadata=None,
        observation_scopes=None,
        entities=[],
        causal_relations=[],
        occurred_start=None,
        occurred_end=None,
        mentioned_at=None,
    )
    unit_ids = await store.insert_facts(
        conn=conn, ops=memory._backend.ops, bank_id=bank_id, facts=[fact], document_id=None
    )
    # Consolidated baseline, not a backlog: a pending fact would defer the unsay.
    await store.mark_consolidated(
        conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=unit_ids, when=datetime.now(timezone.utc)
    )
    return uuid.UUID(unit_ids[0])


# Two paragraph blocks, not one bullet list: the unsay pass targets a block, so the
# retired claim needs to be its own block for a removal to leave the rest standing.
_PAGE_CONTENT = "## Conventions\n\nAutocommit is disabled so changes can be reviewed.\n\nTests run on every push.\n"


async def _create_page(
    memory: MemoryEngine,
    conn,
    bank_id: str,
    request_context: RequestContext,
    *,
    based_on: dict[str, list[dict[str, Any]]],
    content: str = _PAGE_CONTENT,
) -> str:
    """A delta-mode page with a stored grounding, as a real refresh would leave it."""
    model = await memory.create_mental_model(
        bank_id=bank_id,
        name="Conventions",
        source_query="What are the project's conventions?",
        content=content,
        request_context=request_context,
        trigger={"mode": "delta", "refresh_after_consolidation": True},
    )
    # Stamp the grounding and the watermark a real refresh would have left behind.
    await conn.execute(
        f"UPDATE {fq_table('mental_models')} SET content = $3, reflect_response = $4::jsonb, "
        f"last_memory_seen_at = now() WHERE bank_id = $1 AND id = $2",
        bank_id,
        model["id"],
        content,
        json.dumps({"text": content, "based_on": based_on, "mental_models": []}),
    )
    return model["id"]


async def _staleness_row(conn, bank_id: str, mental_model_id: str):
    return await conn.fetchrow(
        f"SELECT id, tags, trigger, last_refreshed_at, last_memory_seen_at "
        f"FROM {fq_table('mental_models')} WHERE bank_id = $1 AND id = $2",
        bank_id,
        mental_model_id,
    )


@pytest.fixture
def patch_reflect(monkeypatch):
    """Return a canned reflect result, so these tests exercise the refresh, not recall."""
    from hindsight_api.engine.response_models import ReflectResult

    def _install(memory: MemoryEngine, *, text: str, facts: list[dict] | None = None):
        async def fake_reflect_async(**kwargs):
            return ReflectResult.model_validate(
                {
                    "text": text,
                    "based_on": {
                        "observation": facts or [],
                        "world": [],
                        "experience": [],
                        "mental-models": [],
                        "directives": [],
                    },
                }
            )

        monkeypatch.setattr(memory, "reflect_async", fake_reflect_async)

    return _install


# ---------------------------------------------------------------------------
# Store: which cited ids are still live
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_memory_ids_reports_only_rows_that_exist(memory: MemoryEngine, request_context: RequestContext):
    bank_id = f"test-retract-live-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        live = await _insert_memory(memory, conn, bank_id, "Autocommit is disabled.")
        answered = await get_memories().live_memory_ids(
            conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[str(live), str(uuid.uuid4())]
        )
    assert answered == {str(live)}
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_live_memory_ids_treats_unparseable_ids_as_absent(memory: MemoryEngine, request_context: RequestContext):
    """A based_on may carry ids that are not memory ids at all; they read as gone."""
    bank_id = f"test-retract-unparseable-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        answered = await get_memories().live_memory_ids(
            conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=["coding-style", ""]
        )
    assert answered == set()
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_live_memory_ids_does_not_cross_banks(memory: MemoryEngine, request_context: RequestContext):
    """Bank isolation is strict — another bank's live row must not answer here."""
    bank_id = f"test-retract-iso-a-{uuid.uuid4().hex[:8]}"
    other_bank = f"test-retract-iso-b-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    await _ensure_bank(memory, other_bank, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        elsewhere = await _insert_memory(memory, conn, other_bank, "Someone else's fact.")
        answered = await get_memories().live_memory_ids(
            conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[str(elsewhere)]
        )
    assert answered == set()
    await memory.delete_bank(bank_id, request_context=request_context)
    await memory.delete_bank(other_bank, request_context=request_context)


# ---------------------------------------------------------------------------
# Staleness: a removed memory raises no watermark
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_page_is_not_stale_while_its_grounding_is_intact(memory: MemoryEngine, request_context: RequestContext):
    bank_id = f"test-retract-fresh-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        live = await _insert_memory(memory, conn, bank_id, "Autocommit is disabled.")
        mm_id = await _create_page(
            memory, conn, bank_id, request_context, based_on={"world": [_fact(str(live), "autocommit disabled")]}
        )
        row = await _staleness_row(conn, bank_id, mm_id)
        assert await memory.compute_mental_model_is_stale(conn, bank_id, row) is False
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_invalidating_a_cited_fact_makes_the_page_stale(memory: MemoryEngine, request_context: RequestContext):
    """The gap this closes: invalidation moves the row out, raising no watermark.

    ``any_memory_updated_since`` used to be the only signal, and it is phrased over
    rows that still exist — so neither the consolidation trigger nor the cron loop
    could ever see a retraction, and the page went on stating the retired fact.
    """
    bank_id = f"test-retract-inval-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        live = await _insert_memory(memory, conn, bank_id, "Autocommit is disabled.")
        mm_id = await _create_page(
            memory, conn, bank_id, request_context, based_on={"world": [_fact(str(live), "autocommit disabled")]}
        )

    with patch.object(memory, "_submit_refreshes_for_retracted_grounding", new=AsyncMock(return_value=0)):
        await memory.update_memory_unit(
            bank_id=bank_id, memory_id=str(live), state="invalidated", request_context=request_context
        )

    async with pool.acquire() as conn:
        row = await _staleness_row(conn, bank_id, mm_id)
        assert await memory.compute_mental_model_is_stale(conn, bank_id, row) is True
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_deleting_a_cited_fact_makes_the_page_stale(memory: MemoryEngine, request_context: RequestContext):
    """Deleted and invalidated are not distinguished — neither is true any more."""
    bank_id = f"test-retract-del-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        live = await _insert_memory(memory, conn, bank_id, "Autocommit is disabled.")
        mm_id = await _create_page(
            memory, conn, bank_id, request_context, based_on={"world": [_fact(str(live), "autocommit disabled")]}
        )

    with patch.object(memory, "_submit_refreshes_for_retracted_grounding", new=AsyncMock(return_value=0)):
        await memory.delete_memory_unit(str(live), bank_id=bank_id, request_context=request_context)

    async with pool.acquire() as conn:
        row = await _staleness_row(conn, bank_id, mm_id)
        assert await memory.compute_mental_model_is_stale(conn, bank_id, row) is True
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_grounding_on_another_mental_model_is_never_read_as_retracted(
    memory: MemoryEngine, request_context: RequestContext
):
    """A sibling document's id is not a memory id; it must not look like a removal."""
    bank_id = f"test-retract-sibling-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        mm_id = await _create_page(
            memory,
            conn,
            bank_id,
            request_context,
            based_on={"mental-models": [_fact(str(uuid.uuid4()), "a sibling page", "mental-models")]},
        )
        row = await _staleness_row(conn, bank_id, mm_id)
        assert await memory.compute_mental_model_is_stale(conn, bank_id, row) is False
    await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# The retraction sweep: who gets nudged when a memory disappears
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
# Retracts the grounding with a raw DELETE FROM memory_units. A store-owned bank keeps its
# memories elsewhere, so the delete removes nothing and the precondition never holds.
@pytest.mark.memory_backend_incompatible
async def test_sweep_schedules_a_refresh_for_a_page_whose_grounding_is_gone(
    memory: MemoryEngine, request_context: RequestContext
):
    bank_id = f"test-retract-sweep-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        live = await _insert_memory(memory, conn, bank_id, "Autocommit is disabled.")
        mm_id = await _create_page(
            memory, conn, bank_id, request_context, based_on={"world": [_fact(str(live), "autocommit disabled")]}
        )
        await conn.execute(f"DELETE FROM {fq_table('memory_units')} WHERE bank_id = $1", bank_id)

    submit = AsyncMock(return_value={"operation_id": "op-1"})
    with patch.object(memory, "submit_async_refresh_mental_model", new=submit):
        assert await memory._submit_refreshes_for_retracted_grounding(bank_id, request_context=request_context) == 1
    assert submit.await_args.kwargs["mental_model_id"] == mm_id
    # A bulk curation pass retracts many facts; one queued refresh covers them all,
    # because the set is recomputed when it runs.
    assert submit.await_args.kwargs["skip_if_in_flight"] is True
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_sweep_leaves_manually_refreshed_pages_alone(memory: MemoryEngine, request_context: RequestContext):
    """A page with no auto-refresh has an owner deciding when it runs."""
    bank_id = f"test-retract-manual-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        live = await _insert_memory(memory, conn, bank_id, "Autocommit is disabled.")
        mm_id = await _create_page(
            memory, conn, bank_id, request_context, based_on={"world": [_fact(str(live), "autocommit disabled")]}
        )
        await conn.execute(
            f"UPDATE {fq_table('mental_models')} SET trigger = $3::jsonb WHERE bank_id = $1 AND id = $2",
            bank_id,
            mm_id,
            json.dumps({"mode": "delta", "refresh_after_consolidation": False}),
        )
        await conn.execute(f"DELETE FROM {fq_table('memory_units')} WHERE bank_id = $1", bank_id)

    submit = AsyncMock(return_value={"operation_id": "op-1"})
    with patch.object(memory, "submit_async_refresh_mental_model", new=submit):
        assert await memory._submit_refreshes_for_retracted_grounding(bank_id, request_context=request_context) == 0
    submit.assert_not_awaited()
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_sweep_defers_to_consolidation_when_facts_are_pending(
    memory: MemoryEngine, request_context: RequestContext
):
    """Consolidation is about to run and its completion hook checks the same thing.

    Scheduling here would only queue a refresh that defers on those pending facts
    and has to be repaid afterwards.
    """
    bank_id = f"test-retract-pending-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        live = await _insert_memory(memory, conn, bank_id, "Autocommit is disabled.")
        await _create_page(
            memory, conn, bank_id, request_context, based_on={"world": [_fact(str(live), "autocommit disabled")]}
        )
        await conn.execute(f"DELETE FROM {fq_table('memory_units')} WHERE bank_id = $1 AND id = $2", bank_id, live)
        # An unconsolidated fact left behind by the same edit.
        pending = await _insert_memory(memory, conn, bank_id, "Autocommit is enabled again.")
        await conn.execute(
            f"UPDATE {fq_table('memory_units')} SET consolidated_at = NULL WHERE bank_id = $1 AND id = $2",
            bank_id,
            pending,
        )

    submit = AsyncMock(return_value={"operation_id": "op-1"})
    with patch.object(memory, "submit_async_refresh_mental_model", new=submit):
        assert await memory._submit_refreshes_for_retracted_grounding(bank_id, request_context=request_context) == 0
    submit.assert_not_awaited()
    await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# The unsay pass: a delta refresh removing what a retracted fact supported
# ---------------------------------------------------------------------------


# Blocks are addressed by id, and the ids only exist in the document the model is
# shown. A canned op therefore cannot name one up front — so it names this marker
# and the fake resolves it out of the prompt, which is exactly what a real model
# does. Hardcoding a position instead is the defect the id addressing removed.
_FIRST_BLOCK = "<first-block-of-section>"


def _remove_first_block_of(section_id: str) -> dict[str, Any]:
    """A retraction op removing that section's first block, resolved at call time."""
    return {"op": "remove_block", "section_id": section_id, "block_id": _FIRST_BLOCK}


def _resolve_block_markers(ops: list[dict[str, Any]], user_prompt: str) -> list[dict[str, Any]]:
    """Swap ``_FIRST_BLOCK`` for the real id from the document in the prompt."""
    import json as _json
    import re as _re

    match = _re.search(r"```json\n(.*?)\n```", user_prompt, _re.DOTALL)
    document = _json.loads(match.group(1)) if match else {"sections": []}
    first_block: dict[str, str] = {}
    for section in document.get("sections", []):
        blocks = section.get("blocks") or []
        if blocks:
            first_block[section["id"]] = blocks[0]["id"]

    resolved: list[dict[str, Any]] = []
    for op in ops:
        if op.get("block_id") == _FIRST_BLOCK:
            op = {**op, "block_id": first_block.get(op["section_id"], "missing-block")}
        resolved.append(op)
    return resolved


def _patch_op_calls(monkeypatch, memory: MemoryEngine, *, retraction_ops, delta_ops):
    """Route the two op-generating LLM calls by their system prompt.

    A delta refresh now makes two: the unsay pass and the new-facts pass. They are
    told apart the same way the pipeline distinguishes them — by which system prompt
    they carry — so a test can assert on each independently.
    """
    from hindsight_api.engine.reflect.delta_ops import DeltaOperationList
    from hindsight_api.engine.reflect.prompts import STRUCTURED_RETRACTION_SYSTEM_PROMPT

    calls: list[dict[str, Any]] = []

    async def fake_call(*, messages, **kwargs):
        system = messages[0]["content"]
        is_retraction = system == STRUCTURED_RETRACTION_SYSTEM_PROMPT
        calls.append({"kind": "retraction" if is_retraction else "delta", "messages": messages, **kwargs})
        ops = _resolve_block_markers(retraction_ops if is_retraction else delta_ops, messages[1]["content"])
        return DeltaOperationList.model_validate({"operations": ops})

    monkeypatch.setattr(memory._reflect_llm_config, "call", fake_call)
    return calls


@pytest.mark.asyncio
async def test_delta_refresh_removes_content_resting_on_a_retracted_fact(
    memory: MemoryEngine, request_context: RequestContext, monkeypatch, patch_reflect
):
    """The core repair: the retired claim leaves the page, and stops being cited."""
    bank_id = f"test-retract-unsay-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    gone_id = str(uuid.uuid4())

    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        live = await _insert_memory(memory, conn, bank_id, "Tests run on every push.")
        mm_id = await _create_page(
            memory,
            conn,
            bank_id,
            request_context,
            based_on={
                "world": [
                    _fact(gone_id, "Autocommit was disabled for hunk-level review.", "world"),
                    _fact(str(live), "Tests run on every push.", "world"),
                ]
            },
        )

    patch_reflect(memory, text="No new information.")
    calls = _patch_op_calls(
        monkeypatch,
        memory,
        retraction_ops=[_remove_first_block_of("conventions")],
        delta_ops=[],
    )

    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )

    assert refreshed is not None
    assert "Autocommit is disabled" not in refreshed["content"]
    assert "Tests run on every push" in refreshed["content"], "unrelated content must survive"

    # The retraction pass ran, and it was given the fact's text — the row is gone, so
    # the copy the document recorded is the only thing that could be quoted.
    retraction_call = next(call for call in calls if call["kind"] == "retraction")
    assert "Autocommit was disabled for hunk-level review." in retraction_call["messages"][1]["content"]

    # And the citation is gone, so nothing reports a memory that no longer exists.
    stored = await memory.get_mental_model(bank_id, mm_id, detail="full", request_context=request_context)
    cited = based_on_fact_ids(stored["reflect_response"]["based_on"])
    assert gone_id not in cited

    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_unsay_runs_even_when_no_new_facts_arrived(
    memory: MemoryEngine, request_context: RequestContext, monkeypatch, patch_reflect
):
    """A retraction is a reason to edit the document all by itself.

    Gating the edit on new facts arriving is exactly the coupling that let a retired
    claim survive indefinitely on a bank that had gone quiet.
    """
    bank_id = f"test-retract-quiet-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    gone_id = str(uuid.uuid4())

    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        live = await _insert_memory(memory, conn, bank_id, "Tests run on every push.")
        mm_id = await _create_page(
            memory,
            conn,
            bank_id,
            request_context,
            based_on={
                "world": [_fact(gone_id, "Retired.", "world"), _fact(str(live), "Tests run on every push.", "world")]
            },
        )

    # No facts at all: the refresh would previously have returned
    # content_preserved_no_new_facts without looking at the document.
    patch_reflect(memory, text="", facts=[])
    calls = _patch_op_calls(
        monkeypatch,
        memory,
        retraction_ops=[_remove_first_block_of("conventions")],
        delta_ops=[],
    )

    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )

    assert refreshed is not None
    assert [call["kind"] for call in calls] == ["retraction"], "no new-facts pass, but the unsay must still run"
    assert "Autocommit is disabled" not in refreshed["content"]
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
# Establishes 'pending consolidation' with a raw UPDATE of memory_units.consolidated_at,
# which a store-owned bank does not read, so the deferral precondition is never set up.
@pytest.mark.memory_backend_incompatible
async def test_unsay_is_deferred_while_facts_are_pending_consolidation(
    memory: MemoryEngine, request_context: RequestContext, monkeypatch, patch_reflect
):
    """Re-ingest safety: the replacements have not been consolidated yet.

    Removing prose here would delete claims that are still true, and because the ids
    leave based_on with it, nothing would ever notice again.
    """
    bank_id = f"test-retract-defer-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    gone_id = str(uuid.uuid4())

    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        live = await _insert_memory(memory, conn, bank_id, "Tests run on every push.")
        mm_id = await _create_page(
            memory,
            conn,
            bank_id,
            request_context,
            based_on={
                "world": [_fact(gone_id, "Retired.", "world"), _fact(str(live), "Tests run on every push.", "world")]
            },
        )
        pending = await _insert_memory(memory, conn, bank_id, "Autocommit is enabled again.")
        await conn.execute(
            f"UPDATE {fq_table('memory_units')} SET consolidated_at = NULL WHERE bank_id = $1 AND id = $2",
            bank_id,
            pending,
        )

    patch_reflect(memory, text="No new information.")
    calls = _patch_op_calls(
        monkeypatch,
        memory,
        retraction_ops=[_remove_first_block_of("conventions")],
        delta_ops=[],
    )

    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )

    assert refreshed is not None
    assert not [call for call in calls if call["kind"] == "retraction"], "the unsay must not run"
    assert "Autocommit is disabled" in refreshed["content"], "content must be preserved while deferred"

    # Critically, the citation is KEPT: it is the only remaining evidence that the
    # prose is unsupported, so pruning it would strand the stale claim.
    stored = await memory.get_mental_model(bank_id, mm_id, detail="full", request_context=request_context)
    assert gone_id in based_on_fact_ids(stored["reflect_response"]["based_on"])

    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_failed_unsay_keeps_the_citation_so_the_next_refresh_retries(
    memory: MemoryEngine, request_context: RequestContext, monkeypatch, patch_reflect
):
    bank_id = f"test-retract-fail-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    gone_id = str(uuid.uuid4())

    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        live = await _insert_memory(memory, conn, bank_id, "Tests run on every push.")
        mm_id = await _create_page(
            memory,
            conn,
            bank_id,
            request_context,
            based_on={
                "world": [_fact(gone_id, "Retired.", "world"), _fact(str(live), "Tests run on every push.", "world")]
            },
        )

    patch_reflect(memory, text="No new information.")

    async def exploding_call(*, messages, **kwargs):
        raise RuntimeError("provider is down")

    monkeypatch.setattr(memory._reflect_llm_config, "call", exploding_call)

    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )

    assert refreshed is not None
    stored = await memory.get_mental_model(bank_id, mm_id, detail="full", request_context=request_context)
    assert gone_id in based_on_fact_ids(stored["reflect_response"]["based_on"])
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_full_mode_needs_no_unsay_pass(
    memory: MemoryEngine, request_context: RequestContext, monkeypatch, patch_reflect
):
    """A full refresh regenerates from live facts and rebuilds based_on wholesale."""
    bank_id = f"test-retract-full-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    gone_id = str(uuid.uuid4())

    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        live = await _insert_memory(memory, conn, bank_id, "Tests run on every push.")
        mm_id = await _create_page(
            memory,
            conn,
            bank_id,
            request_context,
            based_on={
                "world": [_fact(gone_id, "Retired.", "world"), _fact(str(live), "Tests run on every push.", "world")]
            },
        )
        await conn.execute(
            f"UPDATE {fq_table('mental_models')} SET trigger = $3::jsonb WHERE bank_id = $1 AND id = $2",
            bank_id,
            mm_id,
            json.dumps({"mode": "full", "refresh_after_consolidation": True}),
        )

    patch_reflect(memory, text="## Conventions\n\n- Tests run on every push.\n")
    calls = _patch_op_calls(monkeypatch, memory, retraction_ops=[], delta_ops=[])

    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )

    assert refreshed is not None
    assert calls == [], "full mode makes no op calls at all"
    assert "Autocommit is disabled" not in refreshed["content"]
    stored = await memory.get_mental_model(bank_id, mm_id, detail="full", request_context=request_context)
    assert gone_id not in based_on_fact_ids(stored["reflect_response"]["based_on"])
    await memory.delete_bank(bank_id, request_context=request_context)


def test_grounding_that_resolves_to_nothing_is_a_broken_link_not_a_retraction():
    """All citations missing is the signature of a restored bank, not of a retraction.

    Whole-bank transfer carries ``mental_models`` rows verbatim while re-creating the
    memories under fresh ids, so every id a restored page cites is absent on arrival.
    Reading that as a retraction would ask the unsay pass to delete everything the
    page says — irreversibly, on a page that is perfectly correct.
    """
    based_on = {"observation": [_fact("a"), _fact("b"), _fact("c")]}
    retracted = partition_retracted(based_on, live_ids=set())
    assert retracted.unresolvable is True


def test_a_partial_loss_is_a_real_retraction():
    based_on = {"observation": [_fact("a"), _fact("b")]}
    retracted = partition_retracted(based_on, live_ids={"a"})
    assert retracted.unresolvable is False
    assert retracted.ids == {"b"}


def test_intact_grounding_is_never_unresolvable():
    retracted = partition_retracted({"observation": [_fact("a")]}, live_ids={"a"})
    assert retracted.unresolvable is False
    assert not retracted


@pytest.mark.asyncio
async def test_restored_page_keeps_its_content_and_drops_its_dangling_citations(
    memory: MemoryEngine, request_context: RequestContext, monkeypatch, patch_reflect
):
    """End to end: a page whose whole grounding is unresolvable must not be gutted."""
    bank_id = f"test-retract-restored-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)

    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        mm_id = await _create_page(
            memory,
            conn,
            bank_id,
            request_context,
            based_on={"world": [_fact(str(uuid.uuid4()), "carried over from the source bank", "world")]},
        )

    patch_reflect(memory, text="No new information.")
    calls = _patch_op_calls(
        monkeypatch,
        memory,
        retraction_ops=[{"op": "remove_section", "section_id": "conventions"}],
        delta_ops=[],
    )

    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )

    assert refreshed is not None
    assert calls == [], "no unsay pass may run against unresolvable grounding"
    assert "Autocommit is disabled" in refreshed["content"]
    assert "Tests run on every push" in refreshed["content"]

    # The dangling citations do go: nothing should report a fact that is not there.
    stored = await memory.get_mental_model(bank_id, mm_id, detail="full", request_context=request_context)
    assert based_on_fact_ids(stored["reflect_response"]["based_on"]) == []

    await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Real-LLM eval for the retraction prompt
# ---------------------------------------------------------------------------
#
# The stubbed tests above prove the pipeline runs the pass, applies its ops and
# prunes the citations. They cannot prove the thing that actually matters here,
# because the stub decides the answer: that a real model, shown a retracted fact
# and a document, removes what rests on it and leaves everything else alone.
#
# That is the risky half of this feature. The edit is irreversible, and the
# prompt's central instruction ("when in doubt, keep it") is exactly the kind of
# judgement no mock can simulate — so it is judged, not string-matched.

_GEMINI_API_KEY = os.getenv("HINDSIGHT_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_RUN_LLM_EVAL = os.getenv("HINDSIGHT_RUN_GEMINI_EVALS") == "1" and (bool(_GEMINI_API_KEY) or bool(_OPENAI_API_KEY))

pytestmark_llm = pytest.mark.skipif(
    not _RUN_LLM_EVAL,
    reason="Set HINDSIGHT_RUN_GEMINI_EVALS=1 and a Gemini/OpenAI API key to run the retraction eval",
)

_RETRACTION_DOC = """## Change Management

Autocommit is disabled in chezmoi (`autocommit = false`) so changes can be reviewed hunk by hunk.

Every push runs the full test suite before merge.

## Review

Two approvals are required on anything touching the migration tree.
"""


@pytest.fixture
def retraction_llm():
    """A real LLM config for the retraction call.

    The ``memory`` fixture wires MockLLM, which echoes its input — a retraction
    prompt judged through it would pass without a model ever having read it.

    Provider and model come from the environment rather than being pinned here.
    Pinning is how the neighbouring delta evals ended up defaulting to
    ``gemini-2.0-flash``, which the provider has since retired: a hardcoded model
    rots silently because the test is normally skipped.
    """
    from hindsight_api.config import get_config
    from hindsight_api.engine.llm_wrapper import LLMConfig

    config = get_config()
    return LLMConfig(
        provider=config.llm_provider,
        api_key=config.llm_api_key or _GEMINI_API_KEY or _OPENAI_API_KEY or "",
        base_url=config.llm_base_url or "",
        model=config.llm_model,
        # Vertex AI authenticates by project + service account, not an api_key, so a
        # config built from provider/key/model alone raises before any call. The gap
        # was invisible while this test was skipped everywhere; CI runs the evals
        # under `provider=vertexai`, which is exactly the case it did not cover.
        vertexai_project_id=config.llm_vertexai_project_id,
        vertexai_region=config.llm_vertexai_region,
        vertexai_service_account_key=config.llm_vertexai_service_account_key,
    )


@pytestmark_llm
@pytest.mark.hs_llm_core
@pytest.mark.asyncio
async def test_real_model_removes_only_the_retracted_claim(retraction_llm):
    """The prompt's actual contract, against a real model.

    Uses the issue's own example — a setting that was true when written and was
    reversed the next day — because that is the shape the feature exists for.
    """
    from hindsight_api.engine.reflect.delta_ops import apply_operations, parse_delta_operation_list
    from hindsight_api.engine.reflect.prompts import (
        STRUCTURED_RETRACTION_SYSTEM_PROMPT,
        build_structured_retraction_prompt,
    )
    from hindsight_api.engine.reflect.structured_doc import render_document, split_markdown
    from tests.llm_judge import assert_meets_criteria

    document = split_markdown(_RETRACTION_DOC)
    prompt = build_structured_retraction_prompt(
        current_document_json=document.model_dump_json(),
        retracted_facts=[
            {
                "id": str(uuid.uuid4()),
                "text": "Autocommit was disabled in chezmoi (autocommit = false) to enable hunk-level review.",
                "type": "world",
                "context": None,
            }
        ],
        surviving_facts=[
            {
                "id": str(uuid.uuid4()),
                "text": "Two approvals are required on changes to the migration tree.",
                "type": "world",
                "context": None,
            }
        ],
        source_query="What are the project's change-management conventions?",
        max_output_tokens=2048,
    )

    raw = await retraction_llm.call(
        messages=[
            {"role": "system", "content": STRUCTURED_RETRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=4096,
        temperature=0.0,
        scope="mental_model_retraction_ops",
    )
    outcome = apply_operations(document, parse_delta_operation_list(raw).operations)
    result = render_document(outcome.document)

    await assert_meets_criteria(
        response=result,
        criteria=(
            "The document no longer states that autocommit is disabled, or that changes are reviewed "
            "hunk by hunk because autocommit is off. It still states that every push runs the full "
            "test suite, and that two approvals are required for the migration tree. It contains no "
            "note, placeholder, or remark saying that anything was removed or retracted."
        ),
        context=(
            "This is a change-management document after a pass that was asked to remove content resting "
            "on one retracted fact: that autocommit was disabled in chezmoi for hunk-level review. The "
            "other two statements were not retracted and must survive untouched."
        ),
    )
