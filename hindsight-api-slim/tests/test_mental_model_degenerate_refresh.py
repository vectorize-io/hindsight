"""A degenerate but non-empty reflect answer must not overwrite a real document (#4860).

#2959's fix (#3761) closed the placeholder path by raising on a blank answer, but
the only check left on the write path was still emptiness. A model that collapses
under prompt growth can return two tokens — ``OK`` — and that passed the guard and
replaced a multi-thousand-character document, with the operation reporting success.

The guard here is on the WRITE (in the executor, after the delta-not-applied
refusal), not in the reflect agent: any future degenerate-answer path would still
reach the persist path, so the safety net has to sit where the write happens.

Like ``empty_candidate``, the refusal routes through ``_preserve_and_fail``: the
previous content, structure and watermark all stand, ``refresh_skipped`` records
the reason, the model's history gains a failure row, and the caller sees a raise —
so a scheduled refresh fails loudly instead of silently destroying the document.
"""

import dataclasses
import os
import uuid

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.memory_engine import MemoryEngine, MentalModelRefreshError
from tests.conftest import stub_refresh_has_sources

EXISTING = (
    "# Team\n\nAlice leads the platform team. Bob owns data ingest.\n"
    "Carol runs QA. Dave handles the customer escalations rota.\n"
)


async def _model_with_content(
    memory: MemoryEngine, request_context: RequestContext, bank_id: str, *, trigger: dict | None = None
) -> dict:
    await memory.ensure_bank_profile(bank_id, request_context=request_context)
    return await memory.create_mental_model(
        bank_id=bank_id,
        name="Team Info",
        source_query="Tell me about the team",
        content=EXISTING,
        trigger=trigger or {"mode": "full"},
        request_context=request_context,
    )


def _patch_reflect(memory: MemoryEngine, monkeypatch, *, text: str, fact_ids: list[str] | None = None) -> None:
    """Make reflect_async return a canned answer.

    ``fact_ids`` are synthetic; they only feed the delta supporting-facts payload
    and the stored ``based_on``. The refresh that STORES them is the last one the
    test runs, so nothing ever re-checks those ids against live memories (the
    retraction pass reads the PREVIOUSLY stored based_on, which stays empty).
    """

    async def canned_reflect(**kwargs):
        from hindsight_api.engine.response_models import ReflectResult

        return ReflectResult.model_validate(
            {
                "text": text,
                "based_on": {
                    "observation": [
                        {"id": fid, "text": "some fact", "type": "observation", "context": None}
                        for fid in (fact_ids or [])
                    ],
                    "world": [],
                    "experience": [],
                    "mental-models": [],
                    "directives": [],
                },
            }
        )

    monkeypatch.setattr(memory, "reflect_async", canned_reflect)
    stub_refresh_has_sources(monkeypatch, memory)


@pytest.mark.asyncio
async def test_degenerate_answer_does_not_overwrite_full_document(memory, request_context, monkeypatch):
    """The reported case: reflect answers ``OK`` over a long document; the guard refuses it."""
    bank_id = f"test-mm-degenerate-{uuid.uuid4().hex[:8]}"
    try:
        mm = await _model_with_content(memory, request_context, bank_id)
        _patch_reflect(memory, monkeypatch, text="OK", fact_ids=["obs-1"])

        with pytest.raises(MentalModelRefreshError) as excinfo:
            await memory.refresh_mental_model(
                bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
            )

        assert excinfo.value.reason == "candidate_too_short"
        assert excinfo.value.outcome == "refresh_failed_candidate_too_short"

        preserved = await memory.get_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        assert preserved is not None
        assert preserved["content"] == EXISTING, "A degenerate reflect answer overwrote the stored document (#4860)"
        rr = preserved.get("reflect_response") or {}
        assert rr.get("refresh_skipped") == "candidate_too_short"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_degenerate_answer_does_not_advance_watermark_or_refresh_time(memory, request_context, monkeypatch):
    """The failed run must stay repeatable: the watermark and last_refreshed_at stand."""
    bank_id = f"test-mm-degenerate-wm-{uuid.uuid4().hex[:8]}"
    try:
        mm = await _model_with_content(memory, request_context, bank_id)
        before = await memory.get_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        _patch_reflect(memory, monkeypatch, text="Done.")

        with pytest.raises(MentalModelRefreshError):
            await memory.refresh_mental_model(
                bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
            )

        after = await memory.get_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        assert after is not None and before is not None
        assert after["last_refreshed_at"] == before["last_refreshed_at"]
        assert after["last_memory_seen_at"] == before["last_memory_seen_at"]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_degenerate_answer_is_recorded_in_model_history(memory, request_context, monkeypatch):
    """The failure reaches the model's own history, not only the raised exception."""
    bank_id = f"test-mm-degenerate-hist-{uuid.uuid4().hex[:8]}"
    try:
        mm = await _model_with_content(memory, request_context, bank_id)
        _patch_reflect(memory, monkeypatch, text="OK", fact_ids=["obs-1"])

        with pytest.raises(MentalModelRefreshError):
            await memory.refresh_mental_model(
                bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
            )

        history = await memory.get_mental_model_history(bank_id, mm["id"], request_context=request_context)
        assert history is not None
        failure_rows = [entry for entry in history if entry.get("kind") == "refresh_failed"]
        assert failure_rows, "the refused write left no failure row in the model's history"
        latest = failure_rows[0]
        assert latest["outcome"] == "refresh_failed_candidate_too_short"
        assert latest["failure_reason"] == "candidate_too_short"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_full_substantive_answer_still_writes(memory, request_context, monkeypatch):
    """A real synthesis over the same document must not be refused by the new guard."""
    bank_id = f"test-mm-degenerate-ok-{uuid.uuid4().hex[:8]}"
    try:
        mm = await _model_with_content(memory, request_context, bank_id)

        rewritten = (
            "# Team\n\nAlice leads the platform team. Bob owns data ingest.\n"
            "Carol runs QA. Dave handles the customer escalations rota.\n"
            "Eve joined as the platform team's second engineer this week.\n"
        )
        _patch_reflect(memory, monkeypatch, text=rewritten, fact_ids=["obs-1"])

        await memory.refresh_mental_model(bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context)

        stored = await memory.get_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        assert stored is not None
        assert "Eve joined" in stored["content"], "A substantive refresh was refused by the new guard"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_full_degenerate_answer_over_empty_model_is_allowed(memory, request_context, monkeypatch):
    """No existing content means no baseline to shrink from: the candidate is kept.

    A newly created page stores no content until its first refresh, and the legacy
    placeholder is not a baseline either (see ``has_delta_baseline``) — refusing a
    short first document would leave those pages wedged on empty forever.
    """
    bank_id = f"test-mm-degenerate-empty-{uuid.uuid4().hex[:8]}"
    try:
        await memory.ensure_bank_profile(bank_id, request_context=request_context)
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Fresh Page",
            source_query="Tell me about the team",
            content="",
            trigger={"mode": "full"},
            request_context=request_context,
        )
        _patch_reflect(memory, monkeypatch, text="OK", fact_ids=["obs-1"])

        result = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        assert result is not None
        assert result["content"].strip() != "", "A first refresh over an empty page was refused"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_delta_mode_shrink_is_exempt_from_the_guard(memory, request_context, monkeypatch):
    """The guard is full-mode only: a delta edit that rewrites the render smaller is legitimate.

    A delta that removes a block re-renders a much smaller document — that is the
    feature, not a degenerate answer. Seeding follows the proven two-refresh
    pattern from ``test_mental_model_delta.py``: refresh one records the tracking
    row (delta with an empty supporting-fact set preserves and seeds), refresh two
    carries a real fact so the delta op call fires, and its single ``remove_block``
    op shrinks the render past where any full-mode floor would sit.
    """
    from hindsight_api.engine.llm_wrapper import LLMCallResult
    from hindsight_api.engine.reflect.structured_doc import split_markdown
    from hindsight_api.engine.response_models import TokenUsage

    bank_id = f"test-mm-degenerate-delta-{uuid.uuid4().hex[:8]}"
    try:
        mm = await _model_with_content(memory, request_context, bank_id, trigger={"mode": "delta"})

        # Refresh 1: no supporting facts -> preserves content and seeds the tracking row.
        _patch_reflect(memory, monkeypatch, text="ignored — delta preserves without new facts")
        await memory.refresh_mental_model(bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context)

        # The baseline the second refresh will edit, and the block to remove.
        baseline = split_markdown(EXISTING)
        section = baseline.sections[0]
        victim = section.blocks[-1]

        async def removing_delta_call(*, messages, **kwargs):
            from hindsight_api.engine.reflect.delta_ops import DeltaOperationList, RemoveBlockOp

            ops = DeltaOperationList(operations=[RemoveBlockOp(section_id=section.id, block_id=victim.id)])
            return LLMCallResult(content=ops, usage=TokenUsage())

        monkeypatch.setattr(memory._mental_model_refresh_llm_config, "call", removing_delta_call)

        # Refresh 2: one real fact in based_on so the delta op call fires, and its
        # single op shrinks the document. Must be reported as written, not refused.
        _patch_reflect(memory, monkeypatch, text="OK", fact_ids=["obs-1"])
        result = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        assert result is not None
        assert "Dave handles" not in result["content"], "the remove_block op did not land"
        assert len(result["content"]) < len(EXISTING)
        rr = result.get("reflect_response") or {}
        assert rr.get("outcome") == "content_written"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_dry_run_reports_the_refusal_instead_of_raising(memory, request_context, monkeypatch):
    """A real refresh raises here. The dry run reports it instead, like empty_candidate."""
    bank_id = f"test-mm-degenerate-dry-{uuid.uuid4().hex[:8]}"
    try:
        mm = await _model_with_content(memory, request_context, bank_id)
        _patch_reflect(memory, monkeypatch, text="OK", fact_ids=["obs-1"])

        result = await memory.dry_run_refresh_mental_model(bank_id, mm["id"], request_context=request_context)
        assert result is not None
        assert result.outcome == "refresh_failed_candidate_too_short"
        assert result.would_persist is False
        # The stored document is normalised (trailing newline stripped) by the write
        # path; the preview is what stands in the DB, not the raw create-time text.
        assert result.preview_content == EXISTING.strip()
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_ratio_zero_disables_the_guard(memory, request_context, monkeypatch):
    """HINDSIGHT_API_MENTAL_MODEL_REFRESH_MIN_CONTENT_RATIO=0 restores the old behaviour."""
    from hindsight_api.config import clear_config_cache

    bank_id = f"test-mm-degenerate-off-{uuid.uuid4().hex[:8]}"
    original = os.environ.get("HINDSIGHT_API_MENTAL_MODEL_REFRESH_MIN_CONTENT_RATIO")
    try:
        mm = await _model_with_content(memory, request_context, bank_id)
        _patch_reflect(memory, monkeypatch, text="OK", fact_ids=["obs-1"])

        os.environ["HINDSIGHT_API_MENTAL_MODEL_REFRESH_MIN_CONTENT_RATIO"] = "0"
        clear_config_cache()
        result = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        assert result is not None
        assert result["content"].strip() == "OK", "ratio=0 must disable the shrink guard"
    finally:
        if original is None:
            os.environ.pop("HINDSIGHT_API_MENTAL_MODEL_REFRESH_MIN_CONTENT_RATIO", None)
        else:
            os.environ["HINDSIGHT_API_MENTAL_MODEL_REFRESH_MIN_CONTENT_RATIO"] = original
        clear_config_cache()
        await memory.delete_bank(bank_id, request_context=request_context)


def test_config_rejects_out_of_range_ratio():
    """A ratio outside (0, 1] fails config validation instead of misbehaving at refresh time."""
    from hindsight_api.config import HindsightConfig

    base = HindsightConfig.from_env()
    with_ratio = dataclasses.replace(base, mental_model_refresh_min_content_ratio=1.5)
    with pytest.raises(ValueError, match="mental_model_refresh_min_content_ratio"):
        with_ratio.validate()
