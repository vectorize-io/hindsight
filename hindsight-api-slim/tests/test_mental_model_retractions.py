"""Retracted grounding: facts a document cites that no longer exist.

Split the way the project's testing note asks for. The mechanics here are
deterministic and asserted directly — which ids count as retracted, what gets
pruned, when the unsay pass runs and when it is deferred. What an LLM does with
the retraction prompt is not tested here: the pass is driven through a stub that
returns a fixed operation list, so these cover the pipeline, not the model.
"""

from hindsight_api.engine.response_models import LLMCallResult, TokenUsage
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
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.schema import fq_store_table
from hindsight_api.engine.reflect.retractions import (
    MEMORY_BACKED_FACT_TYPES,
    based_on_fact_ids,
    drop_inline_citations,
    partition_retracted,
    prune_based_on,
)
from hindsight_api.engine.reflect.structured_doc import render_document, split_markdown

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


_A = "11111111-1111-4111-8111-111111111111"
_B = "22222222-2222-4222-8222-222222222222"
_C = "33333333-3333-4333-8333-333333333333"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"Opens at 09:00. [{_A}]", "Opens at 09:00."),
        (f"Opens at 09:00. [{_A}, {_B}]", f"Opens at 09:00. [{_B}]"),
        (f"Opens at 09:00. [{_B}, {_A}, {_C}]", f"Opens at 09:00. [{_B}, {_C}]"),
        (f"Opens at 09:00. [{_A}][{_B}]", f"Opens at 09:00. [{_B}]"),
        (f"- Opens at 09:00 [{_A}]\n- Closes at 17:00 [{_B}]", f"- Opens at 09:00\n- Closes at 17:00 [{_B}]"),
        # Resolved handles land as bare UUIDs, so an id is not always bracketed.
        (f"Opens at 09:00 ({_A}).", "Opens at 09:00."),
        (f"Opens at 09:00 {_A}.", "Opens at 09:00."),
        (f"Opens at 09:00 [[{_A}]].", "Opens at 09:00."),
        (f"See [{_A}](/memories/{_A}).", "See."),
        (f"See [the notes](/memories/{_A}).", "See the notes."),
        (f"{_A}, {_B} and more.", f"{_B} and more."),
        (f"[{_A}] Opens at 09:00.", "Opens at 09:00."),
        (f"Opens at 09:00 [{_A}; {_B}].", f"Opens at 09:00 [{_B}]."),
        (f"Opens at 09:00 (`{_A}`).", "Opens at 09:00."),
        (f"- {_A}\n- Closes at 17:00", "- Closes at 17:00"),
        (f"- Closes at 17:00\n- {_A}", "- Closes at 17:00"),
        (f"- [{_A}] Opens at 09:00", "- Opens at 09:00"),
        (f"1. {_A}\n2. Closes at 17:00", "2. Closes at 17:00"),
        (f"> Opens at 09:00\n> {_A}", "> Opens at 09:00"),
        (f"Note:\n{_A}\nEnd", "Note:\nEnd"),
        (f"Opened in\n2024. {_A}", "Opened in\n2024."),
        (f"Steps:\n1. Open the door\n2. {_A}", "Steps:\n1. Open the door"),
        (f"1. Open the door\n2. {_A}\n3. Close it", "1. Open the door\n3. Close it"),
        # Case does not hide an id, and a group of only dropped ids leaves no ``[]``.
        (f"Opens at 09:00 [{_A.upper()}, {_A}, {_B}].", f"Opens at 09:00 [{_B}]."),
        # A UUID that is not being dropped, and brackets that never held one, stay.
        (f"Ticket {_C} [{_C}] calls foo() and [].", f"Ticket {_C} [{_C}] calls foo() and []."),
        (f"Build {_A}1 is not the id.", f"Build {_A}1 is not the id."),
    ],
)
def test_drop_inline_citations_removes_only_the_dropped_ids(text: str, expected: str):
    doc = split_markdown(f"## Hours\n\n{text}\n")
    cleaned = drop_inline_citations(doc, {_A})
    assert cleaned.sections[0].blocks[0].text == expected
    assert cleaned.sections[0].blocks[0].id == doc.sections[0].blocks[0].id


def test_drop_inline_citations_drops_a_block_that_was_only_the_id():
    doc = split_markdown(f"## Hours\n\nOpens at 09:00.\n\n- {_A}\n")
    cleaned = drop_inline_citations(doc, {_A})
    assert [block.text for block in cleaned.sections[0].blocks] == ["Opens at 09:00."]


def test_drop_inline_citations_without_ids_is_a_no_op():
    doc = split_markdown(f"## Hours\n\nOpens at 09:00. [{_A}]\n")
    assert drop_inline_citations(doc, set()) == doc
    assert render_document(drop_inline_citations(doc, {_B})) == render_document(doc)


# ---------------------------------------------------------------------------
# Fixtures shared by the DB-backed tests
# ---------------------------------------------------------------------------


async def _ensure_bank(memory: MemoryEngine, bank_id: str, request_context: RequestContext) -> None:
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)


async def _insert_memory(memory: MemoryEngine, conn, bank_id: str, text: str) -> uuid.UUID:
    """Seed one memory through the store, bypassing the LLM retain pipeline."""
    store = get_memories()
    fact = SimpleNamespace(
        fact_text=text,
        embedding=(await memory.embeddings.encode([text]))[0],
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
        attachment_ids=[],
    )
    unit_ids = await store.insert_facts(
        conn=conn, ops=memory._backend.ops, bank_id=bank_id, facts=[fact], document_id=None
    )
    # Consolidated baseline, not a backlog: a pending fact would defer the unsay.
    await store.mark_consolidated(
        conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_ids=unit_ids, when=datetime.now(timezone.utc)
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
        f"UPDATE {fq_store_table('mental_models')} SET content = $3, reflect_response = $4::jsonb, "
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
        f"FROM {fq_store_table('mental_models')} WHERE bank_id = $1 AND id = $2",
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
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_ids=[str(live), str(uuid.uuid4())]
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
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_ids=["coding-style", ""]
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
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_ids=[str(elsewhere)]
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
        await conn.execute(f"DELETE FROM {fq_store_table('memory_units')} WHERE bank_id = $1", bank_id)

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
            f"UPDATE {fq_store_table('mental_models')} SET trigger = $3::jsonb WHERE bank_id = $1 AND id = $2",
            bank_id,
            mm_id,
            json.dumps({"mode": "delta", "refresh_after_consolidation": False}),
        )
        await conn.execute(f"DELETE FROM {fq_store_table('memory_units')} WHERE bank_id = $1", bank_id)

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
        await conn.execute(
            f"DELETE FROM {fq_store_table('memory_units')} WHERE bank_id = $1 AND id = $2", bank_id, live
        )
        # An unconsolidated fact left behind by the same edit.
        pending = await _insert_memory(memory, conn, bank_id, "Autocommit is enabled again.")
        await conn.execute(
            f"UPDATE {fq_store_table('memory_units')} SET consolidated_at = NULL WHERE bank_id = $1 AND id = $2",
            bank_id,
            pending,
        )

    submit = AsyncMock(return_value={"operation_id": "op-1"})
    with patch.object(memory, "submit_async_refresh_mental_model", new=submit):
        assert await memory._submit_refreshes_for_retracted_grounding(bank_id, request_context=request_context) == 0
    submit.assert_not_awaited()
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_sweep_schedules_on_a_bank_without_observations_despite_pending_facts(
    memory: MemoryEngine, request_context: RequestContext
):
    """#6855: with observations off nothing is ever consolidated, so "pending" never drains.

    Waiting on it would mean never scheduling the refresh that removes the deleted claim.
    """
    bank_id = f"test-retract-noobs-sweep-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    await memory.update_bank_config(bank_id, {"enable_observations": False}, request_context=request_context)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        live = await _insert_memory(memory, conn, bank_id, "Autocommit is disabled.")
        await _create_page(
            memory, conn, bank_id, request_context, based_on={"world": [_fact(str(live), "autocommit disabled")]}
        )
        await conn.execute(
            f"DELETE FROM {fq_store_table('memory_units')} WHERE bank_id = $1 AND id = $2", bank_id, live
        )
        pending = await _insert_memory(memory, conn, bank_id, "Autocommit is enabled again.")
        await conn.execute(
            f"UPDATE {fq_store_table('memory_units')} SET consolidated_at = NULL WHERE bank_id = $1 AND id = $2",
            bank_id,
            pending,
        )

    submit = AsyncMock(return_value={"operation_id": "op-1"})
    with patch.object(memory, "submit_async_refresh_mental_model", new=submit):
        assert await memory._submit_refreshes_for_retracted_grounding(bank_id, request_context=request_context) == 1
    submit.assert_awaited_once()
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
        return LLMCallResult(content=DeltaOperationList.model_validate({"operations": ops}), usage=TokenUsage())

    monkeypatch.setattr(memory._mental_model_refresh_llm_config, "call", fake_call)
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
async def test_unsay_call_uses_the_mental_model_refresh_temperature(
    memory: MemoryEngine, request_context: RequestContext, monkeypatch, patch_reflect
):
    """The retraction pass sends ``llm_temperature_mental_model_refresh``, like the
    delta pass, rather than the consolidation temperature."""
    from hindsight_api.config import get_config

    monkeypatch.setattr(get_config()._config, "llm_temperature_consolidation", 0.0)
    monkeypatch.setattr(get_config()._config, "llm_temperature_mental_model_refresh", 0.65)

    bank_id = f"test-retract-temperature-{uuid.uuid4().hex[:8]}"
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

    await memory.refresh_mental_model(bank_id=bank_id, mental_model_id=mm_id, request_context=request_context)

    retraction_call = next(call for call in calls if call["kind"] == "retraction")
    assert retraction_call["scope"] == "mental_model_retraction_ops"
    assert retraction_call["temperature"] == 0.65

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
            f"UPDATE {fq_store_table('memory_units')} SET consolidated_at = NULL WHERE bank_id = $1 AND id = $2",
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
async def test_unsay_runs_on_a_bank_without_observations_despite_pending_facts(
    memory: MemoryEngine, request_context: RequestContext, monkeypatch, patch_reflect
):
    """#6855: the consolidation wait has nothing to wait for when observations are off.

    Such a bank never consolidates, so every fact stays pending and the deferral above
    used to hold the deleted claim on the page forever.
    """
    bank_id = f"test-retract-noobs-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    await memory.update_bank_config(bank_id, {"enable_observations": False}, request_context=request_context)
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
            f"UPDATE {fq_store_table('memory_units')} SET consolidated_at = NULL WHERE bank_id = $1 AND id = $2",
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
    assert [call for call in calls if call["kind"] == "retraction"], "the unsay must run"
    assert "Autocommit is disabled" not in refreshed["content"]
    stored = await memory.get_mental_model(bank_id, mm_id, detail="full", request_context=request_context)
    assert gone_id not in based_on_fact_ids(stored["reflect_response"]["based_on"])

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

    monkeypatch.setattr(memory._mental_model_refresh_llm_config, "call", exploding_call)

    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )

    assert refreshed is not None
    stored = await memory.get_mental_model(bank_id, mm_id, detail="full", request_context=request_context)
    assert gone_id in based_on_fact_ids(stored["reflect_response"]["based_on"])
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.parametrize("new_facts_failure", ["all_ops_skipped", "call_raises"])
async def test_failed_new_facts_delta_fails_the_refresh_even_after_an_unsay_edit(
    memory: MemoryEngine, request_context: RequestContext, monkeypatch, patch_reflect, new_facts_failure
):
    """#6447: the unsay pass edited the page, then the new-facts delta failed.

    Writing the unsay-only document would move the watermark past the new facts, so
    no later delta would ever show them to the page. The refresh must fail instead,
    leaving content, citations and watermark as they were, so the next one retries both.
    """
    from hindsight_api.engine.memory_engine import MentalModelRefreshError

    bank_id = f"test-retract-newfail-{uuid.uuid4().hex[:8]}"
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
        # Written after the watermark, so the next delta window contains it.
        new = await _insert_memory(memory, conn, bank_id, "Releases are cut on Fridays.")
        before = await _staleness_row(conn, bank_id, mm_id)

    # A new fact arrived, so the new-facts delta runs after the unsay pass.
    patch_reflect(memory, text="Releases are cut on Fridays.", facts=[_fact(str(new), "Releases are cut on Fridays.")])
    calls = _patch_op_calls(
        monkeypatch,
        memory,
        retraction_ops=[_remove_first_block_of("conventions")],
        delta_ops=[{"op": "append_block", "section_id": "no-such-section", "text": "x"}],
    )
    if new_facts_failure == "call_raises":
        routed = memory._mental_model_refresh_llm_config.call

        async def failing_delta(*, messages, **kwargs):
            from hindsight_api.engine.reflect.prompts import STRUCTURED_RETRACTION_SYSTEM_PROMPT

            if messages[0]["content"] != STRUCTURED_RETRACTION_SYSTEM_PROMPT:
                raise ValueError("model returned garbage")
            return await routed(messages=messages, **kwargs)

        monkeypatch.setattr(memory._mental_model_refresh_llm_config, "call", failing_delta)

    with pytest.raises(MentalModelRefreshError) as failed:
        await memory.refresh_mental_model(bank_id=bank_id, mental_model_id=mm_id, request_context=request_context)
    expected_reason = "delta_ops_all_skipped" if new_facts_failure == "all_ops_skipped" else "delta_ops_failed"
    assert failed.value.reason == expected_reason

    assert any(call["kind"] == "retraction" for call in calls), "the unsay pass must have run"
    stored = await memory.get_mental_model(bank_id, mm_id, detail="full", request_context=request_context)
    assert stored["content"] == _PAGE_CONTENT, "the unsay-only document must not be written"
    assert gone_id in based_on_fact_ids(stored["reflect_response"]["based_on"]), "the retraction must be retried"
    async with pool.acquire() as conn:
        after = await _staleness_row(conn, bank_id, mm_id)
    assert after["last_memory_seen_at"] == before["last_memory_seen_at"], "the watermark must not move"

    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_unsay_and_new_facts_both_land_in_one_refresh(
    memory: MemoryEngine, request_context: RequestContext, monkeypatch, patch_reflect
):
    """The new-facts delta edits the document the unsay pass left, and both are written."""
    bank_id = f"test-retract-both-{uuid.uuid4().hex[:8]}"
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
        before = await _staleness_row(conn, bank_id, mm_id)
        new = await _insert_memory(memory, conn, bank_id, "Releases are cut on Fridays.")

    patch_reflect(memory, text="Releases are cut on Fridays.", facts=[_fact(str(new), "Releases are cut on Fridays.")])
    _patch_op_calls(
        monkeypatch,
        memory,
        retraction_ops=[_remove_first_block_of("conventions")],
        delta_ops=[{"op": "append_block", "section_id": "conventions", "text": "Releases are cut on Fridays."}],
    )

    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )

    assert refreshed is not None
    assert "Autocommit is disabled" not in refreshed["content"], "the unsay edit must survive the new-facts pass"
    assert "Releases are cut on Fridays." in refreshed["content"]
    assert "Tests run on every push" in refreshed["content"]
    stored = await memory.get_mental_model(bank_id, mm_id, detail="full", request_context=request_context)
    assert gone_id not in based_on_fact_ids(stored["reflect_response"]["based_on"])
    async with pool.acquire() as conn:
        after = await _staleness_row(conn, bank_id, mm_id)
    assert after["last_memory_seen_at"] > before["last_memory_seen_at"]

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
            f"UPDATE {fq_store_table('mental_models')} SET trigger = $3::jsonb WHERE bank_id = $1 AND id = $2",
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


def test_grounding_that_resolves_to_nothing_is_unresolvable():
    """Every cited fact missing: the page's whole source is gone, so it is rebuilt."""
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
async def test_page_whose_whole_source_was_deleted_is_emptied(
    memory: MemoryEngine, request_context: RequestContext, monkeypatch, patch_reflect
):
    """Every cited fact is gone and nothing replaced them: the page has nothing left to say.

    It used to keep its prose (and its inline citations) forever, on the theory that
    all-missing meant a restored bank — but import rewrites the ids now (#3833), so
    all-missing means the source was deleted (#6764).
    """
    bank_id = f"test-retract-gone-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    gone_id = str(uuid.uuid4())

    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        mm_id = await _create_page(
            memory,
            conn,
            bank_id,
            request_context,
            based_on={"world": [_fact(gone_id, "Autocommit is disabled.", "world")]},
            content=f"## Conventions\n\nAutocommit is disabled. [{gone_id}]\n",
        )

    patch_reflect(memory, text="")
    calls = _patch_op_calls(monkeypatch, memory, retraction_ops=[], delta_ops=[])

    preview = await memory.dry_run_refresh_mental_model(bank_id, mm_id, request_context=request_context)
    assert preview is not None
    assert preview.effective_mode == "full"
    assert preview.mode_fallback_reason == "grounding_gone"
    # Said as what happened: the page is emptied, not "left as it is".
    assert any("emptied" in warning for warning in preview.warnings), preview.warnings

    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )

    assert refreshed is not None
    assert calls == [], "a rebuild makes no op calls"
    assert refreshed["content"].strip() == ""
    stored = await memory.get_mental_model(bank_id, mm_id, detail="full", request_context=request_context)
    assert based_on_fact_ids(stored["reflect_response"]["based_on"]) == []

    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_a_failed_rebuild_with_sources_in_scope_keeps_the_page(
    memory: MemoryEngine, request_context: RequestContext, monkeypatch, patch_reflect
):
    """Empty is only the answer when nothing is left to read.

    With other facts in scope the rebuild runs reflect, and an empty candidate there
    means reflect failed — small models, exhausted retries, a provider error. Writing
    it would wipe the page; the refresh fails instead and the page stays.
    """
    from hindsight_api.engine.memory_engine import MentalModelRefreshError

    bank_id = f"test-retract-gone-failed-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    gone_id = str(uuid.uuid4())

    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        await _insert_memory(memory, conn, bank_id, "Tests run on every push.")
        mm_id = await _create_page(
            memory,
            conn,
            bank_id,
            request_context,
            based_on={"world": [_fact(gone_id, "Autocommit is disabled.", "world")]},
        )

    patch_reflect(memory, text="")
    _patch_op_calls(monkeypatch, memory, retraction_ops=[], delta_ops=[])

    with pytest.raises(MentalModelRefreshError):
        await memory.refresh_mental_model(bank_id=bank_id, mental_model_id=mm_id, request_context=request_context)

    stored = await memory.get_mental_model(bank_id, mm_id, detail="full", request_context=request_context)
    assert stored["content"] == _PAGE_CONTENT
    # The dead citation stays too: it is what makes the next refresh rebuild again.
    assert gone_id in based_on_fact_ids(stored["reflect_response"]["based_on"])

    # And it does: once reflect works, the page is rebuilt from what is in scope.
    patch_reflect(memory, text="## Conventions\n\nTests run on every push.\n")
    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )
    assert refreshed is not None
    assert "Autocommit is disabled" not in refreshed["content"]
    assert "Tests run on every push." in refreshed["content"]

    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_page_whose_source_was_reingested_is_rebuilt_without_stale_ids(
    memory: MemoryEngine, request_context: RequestContext, monkeypatch, patch_reflect
):
    """The #6764 scenario: the same fact came back under a new id.

    The rebuild writes the page from the replacement, so neither the text nor
    ``based_on`` keeps pointing at the old id.
    """
    bank_id = f"test-retract-reingest-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    old_id = str(uuid.uuid4())

    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        new_id = await _insert_memory(memory, conn, bank_id, "The office opens at 09:00.")
        mm_id = await _create_page(
            memory,
            conn,
            bank_id,
            request_context,
            based_on={"world": [_fact(old_id, "The office opens at 09:00.", "world")]},
            content=f"## Office hours\n\nThe office opens at 09:00. [{old_id}]\n",
        )

    patch_reflect(
        memory,
        text="## Office hours\n\nThe office opens at 09:00.\n",
        facts=[{"id": str(new_id), "text": "The office opens at 09:00.", "type": "observation", "context": None}],
    )
    calls = _patch_op_calls(monkeypatch, memory, retraction_ops=[], delta_ops=[])

    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )

    assert refreshed is not None
    assert calls == [], "a rebuild makes no op calls"
    assert "The office opens at 09:00." in refreshed["content"]
    assert old_id not in refreshed["content"]
    stored = await memory.get_mental_model(bank_id, mm_id, detail="full", request_context=request_context)
    cited = based_on_fact_ids(stored["reflect_response"]["based_on"])
    assert old_id not in cited
    assert str(new_id) in cited

    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_page_whose_whole_source_is_gone_waits_for_pending_consolidation(
    memory: MemoryEngine, request_context: RequestContext, monkeypatch, patch_reflect
):
    """Right after a re-ingest the replacements are not consolidated yet: no rebuild."""
    bank_id = f"test-retract-gone-defer-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    gone_id = str(uuid.uuid4())

    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        mm_id = await _create_page(
            memory,
            conn,
            bank_id,
            request_context,
            based_on={"world": [_fact(gone_id, "Autocommit is disabled.", "world")]},
        )
        pending = await _insert_memory(memory, conn, bank_id, "Autocommit is disabled.")
        await conn.execute(
            f"UPDATE {fq_store_table('memory_units')} SET consolidated_at = NULL WHERE bank_id = $1 AND id = $2",
            bank_id,
            pending,
        )

    patch_reflect(memory, text="No new information.")
    calls = _patch_op_calls(monkeypatch, memory, retraction_ops=[], delta_ops=[])

    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )

    assert refreshed is not None
    assert not [call for call in calls if call["kind"] == "retraction"]
    assert "Autocommit is disabled" in refreshed["content"], "content must be preserved while deferred"
    stored = await memory.get_mental_model(bank_id, mm_id, detail="full", request_context=request_context)
    assert gone_id in based_on_fact_ids(stored["reflect_response"]["based_on"])

    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_unsay_drops_the_retracted_inline_citations_from_kept_text(
    memory: MemoryEngine, request_context: RequestContext, monkeypatch, patch_reflect
):
    """The unsay model never sees ids, so a sentence it keeps still cites the dead fact.

    Pruning ``based_on`` alone left that ``[id]`` in the page with nothing to notice it
    again (#6764); the text loses exactly the ids ``based_on`` loses.
    """
    bank_id = f"test-retract-inline-{uuid.uuid4().hex[:8]}"
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
            content=(
                f"## Conventions\n\nAutocommit is disabled. [{gone_id}]\n\n"
                f"Tests run on every push. [{gone_id}, {live}]\n"
            ),
        )

    patch_reflect(memory, text="No new information.")
    _patch_op_calls(monkeypatch, memory, retraction_ops=[], delta_ops=[])

    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )

    assert refreshed is not None
    assert gone_id not in refreshed["content"]
    assert "Autocommit is disabled." in refreshed["content"], "the model kept it; only the citation goes"
    assert f"Tests run on every push. [{live}]" in refreshed["content"]
    stored = await memory.get_mental_model(bank_id, mm_id, detail="full", request_context=request_context)
    assert gone_id not in based_on_fact_ids(stored["reflect_response"]["based_on"])
    assert gone_id not in json.dumps(stored.get("structured_content") or {})

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

    raw = (
        await retraction_llm.call(
            messages=[
                {"role": "system", "content": STRUCTURED_RETRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=4096,
            temperature=0.0,
            scope="mental_model_retraction_ops",
        )
    ).content
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
