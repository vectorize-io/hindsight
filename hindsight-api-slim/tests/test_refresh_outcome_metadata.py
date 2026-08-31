"""Refresh operations expose semantic outcome fields in result_metadata (#2605).

Retain operations have carried machine-readable outcome metadata since 0.8.x
(``unit_ids_count`` etc.). These tests pin the refresh-side parity: a completed
refresh_mental_model operation must let a monitoring layer distinguish
"refreshed with real content" from "refreshed empty" by reading
``result_metadata`` alone, without a follow-up content fetch.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest

from hindsight_api.api.http import OperationResponse, OperationStatusResponse
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.worker.exceptions import RetryTaskAt
from tests.conftest import stub_refresh_has_sources


@pytest.fixture
async def bank_with_model(memory: MemoryEngine, request_context):
    """Bank with one mental model, unique per test for xdist safety."""
    bank_id = f"test-refresh-meta-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id, request_context=request_context)
    mm = await memory.create_mental_model(
        bank_id=bank_id,
        name="Outcome Meta Model",
        source_query="What outcome fields does refresh expose?",
        content="Original content",
        request_context=request_context,
    )
    yield memory, bank_id, mm
    await memory.delete_bank(bank_id, request_context=request_context)


def _fake_refreshed(content: str, based_on: dict) -> dict:
    """Shape of refresh_mental_model's return value as consumed by the handler."""
    return {
        "content": content,
        "reflect_response": {"text": content, "based_on": based_on, "mental_models": []},
        "source_query": "What outcome fields does refresh expose?",
    }


async def _submit_with_fake_refresh(memory, monkeypatch, bank_id, mm, request_context, refreshed):
    """Submit an async refresh whose reflect outcome is stubbed to `refreshed`.

    The patch must land before submission: the test task backend executes the
    queued task synchronously on submit, so this exercises the real path
    (execute_task -> _handle_refresh_mental_model -> metadata write).
    """

    async def fake_refresh(bank_id, mental_model_id, *, request_context):
        return refreshed

    monkeypatch.setattr(memory, "refresh_mental_model", fake_refresh)
    result = await memory.submit_async_refresh_mental_model(
        bank_id=bank_id,
        mental_model_id=mm["id"],
        request_context=request_context,
    )
    await asyncio.sleep(0.1)
    return result["operation_id"]


@pytest.mark.asyncio
async def test_completed_refresh_enriches_result_metadata(bank_with_model, request_context, monkeypatch):
    """A completed refresh writes content_len / populated_content / based_on_counts."""
    memory, bank_id, mm = bank_with_model
    content = "x" * 120
    based_on = {
        "world": [{"id": "f1"}, {"id": "f2"}, {"id": "f3"}],
        "mental-models": [{"id": "m1"}],
    }

    operation_id = await _submit_with_fake_refresh(
        memory, monkeypatch, bank_id, mm, request_context, _fake_refreshed(content, based_on)
    )

    status = await memory.get_operation_status(
        bank_id=bank_id, operation_id=operation_id, request_context=request_context
    )
    assert status["status"] == "completed"
    meta = status["result_metadata"]

    # Submit-time keys are merged with, not replaced by, the outcome fields:
    # existing consumers join on mental_model_id/name.
    assert meta["mental_model_id"] == mm["id"]
    assert meta["name"] == "Outcome Meta Model"

    assert meta["content_len"] == 120
    assert meta["populated_content"] is True
    assert meta["based_on_counts"] == {"world": 3, "mental-models": 1}


# ---------------------------------------------------------------------------
# What the refresh did with the document (#3274)
# ---------------------------------------------------------------------------
#
# ``result_metadata`` is the only per-refresh record kept indefinitely
# (operation_retention_days defaults to 0 = never prune). Before these fields
# existed it could not say what a refresh did: a full rewrite, a delta edit, a
# run that found nothing to change and a delta that emitted no operations all
# wrote {delta_ops_applied: 0, delta_ops_skipped: 0}, and a preserved document
# reports the length of the content it preserved — so content_len /
# populated_content read identically too. Everything finer lived only in
# ``mental_models.reflect_response``, which the next refresh overwrites.
#
# These drive the real refresh pipeline (only the LLM boundary is stubbed) and
# assert on the persisted operation row.


def _patch_reflect(monkeypatch, memory: MemoryEngine, *, text: str, facts: list[dict] | None = None) -> None:
    """Stub the agentic loop with a canned candidate + evidence set."""
    from hindsight_api.engine.response_models import ReflectResult

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
    stub_refresh_has_sources(monkeypatch, memory)


def _patch_delta_llm(monkeypatch, memory: MemoryEngine, *, returns) -> None:
    """Stub the structured-delta call. ``returns`` is a list of op dicts, or an exception to raise."""
    from hindsight_api.engine.reflect.delta_ops import DeltaOperationList

    async def fake_call(*, messages, **kwargs):
        if isinstance(returns, Exception):
            raise returns
        return DeltaOperationList.model_validate({"operations": returns})

    monkeypatch.setattr(memory._reflect_llm_config, "call", fake_call)


@dataclass
class _RefreshOperationViews:
    """One refresh operation as both API surfaces report it.

    The engine returns plain dicts and the HTTP layer feeds them to the response
    models by keyword, so an engine key that does not match a model field name is
    dropped silently. Building the models here is what pins the two together — the
    typed views are what a client actually receives.
    """

    listed: dict[str, Any]
    status: dict[str, Any]
    listed_model: OperationResponse
    status_model: OperationStatusResponse


async def _refresh_operation_views(memory, bank_id, request_context) -> _RefreshOperationViews:
    """The bank's single refresh operation, as both API surfaces report it."""
    listed = await memory.list_operations(bank_id, task_type="refresh_mental_model", request_context=request_context)
    assert listed["total"] == 1, listed
    row = listed["operations"][0]
    status = await memory.get_operation_status(bank_id=bank_id, operation_id=row["id"], request_context=request_context)
    return _RefreshOperationViews(
        listed=row,
        status=status,
        listed_model=OperationResponse(**row),
        status_model=OperationStatusResponse(**status),
    )


@pytest.fixture
async def delta_bank(memory: MemoryEngine, request_context):
    """Bank with one delta-mode mental model that already has a baseline document."""
    bank_id = f"test-refresh-outcome-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id, request_context=request_context)
    mm = await memory.create_mental_model(
        bank_id=bank_id,
        name="Team Info",
        source_query="Tell me about the team",
        content="# Team\n\nAlice is the lead.\n",
        trigger={"mode": "delta"},
        request_context=request_context,
    )
    yield memory, bank_id, mm
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_preserved_and_rewritten_differ_only_by_outcome(memory: MemoryEngine, request_context, monkeypatch):
    """The two cases the pre-#3274 metadata could not tell apart.

    A delta refresh that found no new facts preserves the document; a full-mode
    refresh rewrites it. Both emit zero delta operations and both report a
    populated document of the same length, so ``outcome`` is the only field that
    separates "nothing changed" from "the whole document was rewritten".
    """
    document = "# Team\n\nAlice is the lead.\n"
    # A genuine rewrite of the same length, so content_len cannot tell the two
    # apart either — same-length-but-different is the point, since identical text
    # would (correctly) report content_unchanged instead.
    rewrite = "# Team\n\nAlice is the boss.\n"
    assert len(rewrite) == len(document)
    metadata: dict[str, dict] = {}

    for mode in ("delta", "full"):
        bank_id = f"test-refresh-outcome-{mode}-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team Info",
            source_query="Tell me about the team",
            content=document,
            trigger={"mode": mode},
            request_context=request_context,
        )
        # No facts: delta short-circuits to "nothing to change", while full mode
        # writes its candidate regardless. Same length, so content_len matches.
        _patch_reflect(monkeypatch, memory, text=rewrite, facts=[])
        _patch_delta_llm(monkeypatch, memory, returns=[])

        await memory.submit_async_refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        await asyncio.sleep(0.1)
        views = await _refresh_operation_views(memory, bank_id, request_context)
        metadata[mode] = views.status["result_metadata"]
        await memory.delete_bank(bank_id, request_context=request_context)

    preserved, rewritten = metadata["delta"], metadata["full"]

    # Everything the old metadata carried is identical between the two...
    for field_name in ("content_len", "populated_content", "delta_ops_applied", "delta_ops_skipped"):
        assert preserved[field_name] == rewritten[field_name], field_name

    # ...and only the outcome tells them apart.
    assert preserved["outcome"] == "content_preserved_no_new_facts"
    assert rewritten["outcome"] == "content_written"
    assert "failure_reason" not in preserved
    assert "failure_reason" not in rewritten


def test_operation_outcome_is_a_superset_of_executor_outcome():
    """The two outcome vocabularies must not drift apart.

    ``RefreshOperationOutcome`` is spelled out rather than unioned (a union of
    Literals renders as an anyOf of two enums in the OpenAPI schema), so nothing
    but this test stops a value added to one from being forgotten in the other.
    """
    from typing import get_args

    from hindsight_api.engine.mental_model_refresh import RefreshOperationOutcome, RefreshOutcome

    executor = set(get_args(RefreshOutcome))
    operation = set(get_args(RefreshOperationOutcome))
    assert executor <= operation, f"executor outcomes missing from the operation vocabulary: {executor - operation}"
    # The persist path is the only source of the extra values; if that changes,
    # the comment on RefreshOperationOutcome needs to change with it.
    assert operation - executor == {"refresh_failed_structured_output"}


def test_unknown_outcome_reports_no_details_instead_of_raising():
    """A value this build has no name for must not take the operations list down.

    This runs per row, so a raise would 500 the whole page over one row. The
    shape that produces it is a rolling upgrade: a worker already writing an
    outcome the API server predates — exactly what this branch created when it
    added ``content_unchanged``.
    """
    from hindsight_api.engine.memory_engine import _operation_details

    assert _operation_details("refresh_mental_model", {"outcome": "an_outcome_a_newer_build_writes"}) is None
    assert (
        _operation_details(
            "refresh_mental_model",
            {"outcome": "refresh_failed_delta_not_applied", "failure_reason": "a_reason_a_newer_build_writes"},
        )
        is None
    )


def test_non_refresh_operations_report_no_refresh_details():
    """``details`` is keyed by operation type, so another type's row must not carry a refresh shape."""
    from hindsight_api.engine.memory_engine import _operation_details

    assert _operation_details("batch_retain", {"outcome": "content_written"}) is None
    # A refresh that has not finished has nothing to report yet.
    assert _operation_details("refresh_mental_model", {"mental_model_id": "mm-1"}) is None


def test_delta_failure_reason_narrows_the_fallback_vocabulary():
    """Only reasons a delta can hit *after* being chosen become failure reasons.

    ``no_baseline_content`` / ``source_query_changed`` turn delta off before it
    runs, so a refresh carrying them is a legitimate full regeneration, not a
    failure — they must degrade to the generic value rather than widen the
    failure enum with values it can never mean. The ``None`` case is defensive:
    every branch reaching the delta-not-applied guard sets a reason today.
    """
    from hindsight_api.engine.memory_engine import _delta_failure_reason

    assert _delta_failure_reason("structured_doc_unreadable") == "structured_doc_unreadable"
    assert _delta_failure_reason("delta_ops_failed") == "delta_ops_failed"
    assert _delta_failure_reason("delta_ops_all_skipped") == "delta_ops_all_skipped"
    assert _delta_failure_reason("no_baseline_content") == "delta_not_applied"
    assert _delta_failure_reason("source_query_changed") == "delta_not_applied"
    assert _delta_failure_reason(None) == "delta_not_applied"


# ---------------------------------------------------------------------------
# Every outcome × failure reason, driven off mocked LLM behaviour
# ---------------------------------------------------------------------------

# The document each scenario starts from. It round-trips through
# split_markdown -> render_document unchanged, which is what lets the zero-op
# delta case below be byte-identical rather than merely equivalent.
_BASELINE = "# Team\n\nAlice is the lead.\n"

_VALID_OP = {
    "op": "append_block",
    "section_id": "team",
    "text": "Bob joined the team.",
}
_UNKNOWN_SECTION_OP = {
    "op": "append_block",
    "section_id": "does-not-exist",
    "text": "Bob joined the team.",
}
_FACTS = [{"id": "obs-new", "text": "Bob joined", "type": "observation", "context": None}]
_SCHEMA = {"type": "object", "properties": {"summary": {"type": "string"}}}


@dataclass
class _OutcomeCase:
    """One refresh, described by what the LLMs do and what it must be called.

    The point of the table is that the outcome is decided by observable effect,
    not by which branch produced it — several rows reach the same branch and are
    named differently, and two rows reach different branches and share a name.
    """

    id: str
    mode: str
    reflect_text: str
    expect_outcome: str
    why: str
    facts: list[dict] = field(default_factory=lambda: list(_FACTS))
    delta_returns: Any = field(default_factory=list)
    unparseable_baseline: bool = False
    response_schema: dict | None = None
    structured_output_fails: bool = False
    expect_failure_reason: str | None = None
    expect_ops_applied: int = 0
    expect_ops_skipped: int = 0


_OUTCOME_CASES = [
    _OutcomeCase(
        id="full_rewrite",
        mode="full",
        reflect_text="# Team\n\nAlice leads; Bob joined.\n",
        expect_outcome="content_written",
        why="full mode writes its candidate, and this one differs from the stored document",
    ),
    _OutcomeCase(
        id="full_regenerates_identical_text",
        mode="full",
        reflect_text=_BASELINE,
        expect_outcome="content_unchanged",
        why="a regeneration that reproduces the stored text changed nothing, however it got there",
    ),
    _OutcomeCase(
        id="delta_applies_ops",
        mode="delta",
        reflect_text="# Team\n\nNarrow candidate.\n",
        delta_returns=[_VALID_OP],
        expect_outcome="content_written",
        expect_ops_applied=1,
        why="an op landed, so the document really is different",
    ),
    _OutcomeCase(
        id="delta_emits_no_ops",
        mode="delta",
        reflect_text="# Team\n\nA candidate the model decides adds nothing.\n",
        delta_returns=[],
        expect_outcome="content_unchanged",
        why=(
            "zero ops is not a rejection, so this lands in the *applied* path and re-renders the "
            "document byte-identically — it used to report content_written, indistinguishable "
            "from a real rewrite"
        ),
    ),
    _OutcomeCase(
        id="delta_applies_some_and_rejects_others",
        mode="delta",
        reflect_text="# Team\n\nNarrow candidate.\n",
        delta_returns=[_VALID_OP, _UNKNOWN_SECTION_OP],
        expect_outcome="content_written",
        expect_ops_applied=1,
        expect_ops_skipped=1,
        why="a partial apply still changes the document; the rejected op is recorded, not fatal",
    ),
    _OutcomeCase(
        id="delta_window_empty",
        mode="delta",
        reflect_text="# Team\n\nUnused candidate.\n",
        facts=[],
        expect_outcome="content_preserved_no_new_facts",
        why="no facts at all means the delta LLM is never called — distinct from reading facts and changing nothing",
    ),
    _OutcomeCase(
        id="empty_candidate",
        mode="delta",
        reflect_text="",
        delta_returns=RuntimeError("simulated invalid JSON from provider"),
        expect_outcome="refresh_failed_empty_candidate",
        expect_failure_reason="empty_candidate",
        why="an empty synthesis is an upstream failure; the guard fires before the delta reason is considered",
    ),
    _OutcomeCase(
        id="delta_op_call_failed",
        mode="delta",
        reflect_text="# Team\n\nNarrow candidate.\n",
        delta_returns=RuntimeError("simulated invalid JSON from provider"),
        expect_outcome="refresh_failed_delta_not_applied",
        expect_failure_reason="delta_ops_failed",
        why="the op call never returned usable JSON — distinct from ops that returned and were rejected",
    ),
    _OutcomeCase(
        id="delta_ops_all_rejected",
        mode="delta",
        reflect_text="# Team\n\nNarrow candidate.\n",
        delta_returns=[_UNKNOWN_SECTION_OP],
        expect_outcome="refresh_failed_delta_not_applied",
        expect_failure_reason="delta_ops_all_skipped",
        why="every op was rejected, so persisting would advance the watermark past facts that never landed",
    ),
    _OutcomeCase(
        id="delta_baseline_unreadable",
        mode="delta",
        reflect_text="# Team\n\nNarrow candidate.\n",
        unparseable_baseline=True,
        expect_outcome="refresh_failed_delta_not_applied",
        expect_failure_reason="structured_doc_unreadable",
        why="delta had no baseline to edit, and the candidate covers only the delta window",
    ),
    _OutcomeCase(
        id="structured_output_extraction_failed",
        mode="full",
        reflect_text="# Team\n\nAlice leads; Bob joined.\n",
        response_schema=_SCHEMA,
        structured_output_fails=True,
        expect_outcome="refresh_failed_structured_output",
        expect_failure_reason="structured_output_failed",
        why="the persist path refuses a document the executor already accepted — the one outcome a dry run cannot reach",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _OUTCOME_CASES, ids=lambda c: c.id)
async def test_refresh_outcome_matrix(case: _OutcomeCase, memory: MemoryEngine, request_context, monkeypatch):
    """Each way a refresh can end reaches the operation record under its own name."""
    bank_id = f"test-outcome-{case.id.replace('_', '-')}-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id, request_context=request_context)
    trigger: dict[str, Any] = {"mode": case.mode}
    if case.response_schema:
        trigger["response_schema"] = case.response_schema
    mm = await memory.create_mental_model(
        bank_id=bank_id,
        name="Team Info",
        source_query="Tell me about the team",
        content=_BASELINE,
        trigger=trigger,
        request_context=request_context,
    )

    _patch_reflect(monkeypatch, memory, text=case.reflect_text, facts=case.facts)
    _patch_delta_llm(monkeypatch, memory, returns=case.delta_returns)
    if case.unparseable_baseline:
        from hindsight_api.engine.reflect import structured_doc

        def unreadable(_stored, _markdown: str):
            raise ValueError("simulated unreadable structured document")

        monkeypatch.setattr(structured_doc, "structured_document_from_stored", unreadable)
    if case.structured_output_fails:
        import types

        from hindsight_api.engine.reflect import agent as reflect_agent

        async def extraction_yields_nothing(answer, response_schema, llm_config, reflect_id, max_tokens=None):
            return types.SimpleNamespace(
                structured_output=None, input_tokens=0, output_tokens=0, cached_tokens=0, thoughts_tokens=0
            )

        monkeypatch.setattr(reflect_agent, "_generate_structured_output", extraction_yields_nothing)

    stored_before = (
        await memory.get_mental_model(bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context)
    )["content"]

    submit = memory.submit_async_refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
    )
    if case.expect_outcome.startswith("refresh_failed"):
        # A failed refresh is retryable, so the task layer re-raises it as
        # RetryTaskAt. The metadata is written before that, on the attempt that
        # failed — which is the whole point of recording it there.
        with pytest.raises(RetryTaskAt):
            await submit
    else:
        await submit
        await asyncio.sleep(0.1)

    views = await _refresh_operation_views(memory, bank_id, request_context)
    details = views.status_model.details
    assert details is not None, f"{case.id}: no details recorded ({case.why})"
    assert details.outcome == case.expect_outcome, f"{case.id}: {case.why}"
    assert details.failure_reason == case.expect_failure_reason, f"{case.id}: {case.why}"

    meta = views.status["result_metadata"]
    assert meta.get("delta_ops_applied", 0) == case.expect_ops_applied, case.id
    assert meta.get("delta_ops_skipped", 0) == case.expect_ops_skipped, case.id

    # The outcome is a claim about the document, so check the document against it.
    stored_after = (
        await memory.get_mental_model(bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context)
    )["content"]
    if case.expect_outcome == "content_written":
        assert stored_after != stored_before, f"{case.id}: reported a write but the document is unchanged"
    else:
        assert stored_after == stored_before, f"{case.id}: reported no write but the document changed"

    await memory.delete_bank(bank_id, request_context=request_context)


def test_outcome_matrix_covers_every_outcome_and_reason():
    """The table above must stay exhaustive as the vocabularies grow.

    A new outcome or failure reason added without a row is a value nothing
    proves the writer ever produces — the failure mode is silence, so it needs
    its own assertion rather than trusting the table to be kept up to date.
    """
    from typing import get_args

    from hindsight_api.engine.mental_model_refresh import RefreshFailureReason, RefreshOperationOutcome

    assert {c.expect_outcome for c in _OUTCOME_CASES} == set(get_args(RefreshOperationOutcome))

    covered_reasons = {c.expect_failure_reason for c in _OUTCOME_CASES} - {None}
    # ``delta_not_applied`` is the defensive default of _delta_failure_reason and
    # is unreachable through the pipeline today — every branch that reaches the
    # delta-not-applied guard sets a more specific reason. It is covered by
    # test_delta_failure_reason_narrows_the_fallback_vocabulary instead.
    assert covered_reasons == set(get_args(RefreshFailureReason)) - {"delta_not_applied"}
