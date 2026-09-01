"""A refresh whose retrieval failed must not touch the stored document (#2894).

When a reflect tool raised, reflect handed the exception text back to the model
as a tool result and let the loop continue. The model answered from whatever it
had — usually nothing — and that answer was indistinguishable from a run over a
bank that genuinely holds nothing on the topic: non-empty prose, so every
emptiness guard on the write path let it through. A mental model built across
months of refreshes was replaced by "I don't have information about that", and
the operation was recorded as completed.

Reflect now fails the run instead (``ReflectToolExecutionError``), so the refresh
never reaches its write. These tests assert that end of it.

The distinction that matters is against a *successful* empty retrieval: a bank
that really has nothing to say on the query must still be allowed to rewrite the
document, so that case is asserted here too.
"""

import uuid

import pytest

from hindsight_api.engine.memory_engine import MemoryEngine, MentalModelRefreshError
from hindsight_api.engine.reflect import ReflectToolExecutionError
from tests.conftest import stub_refresh_has_sources

EXISTING = "# Team\n\nAlice leads platform. Bob owns ingest.\n"


async def _model_with_content(memory: MemoryEngine, request_context, bank_id: str) -> dict:
    await memory.get_bank_profile(bank_id, request_context=request_context)
    return await memory.create_mental_model(
        bank_id=bank_id,
        name="Team Info",
        source_query="Tell me about the team",
        content=EXISTING,
        trigger={"mode": "full"},
        request_context=request_context,
    )


@pytest.mark.asyncio
async def test_refresh_preserves_content_when_a_tool_failed(memory, request_context, monkeypatch):
    """The reported production case: retrieval broke, the model answered anyway."""
    bank_id = f"test-mm-tool-fail-{uuid.uuid4().hex[:8]}"
    try:
        mm = await _model_with_content(memory, request_context, bank_id)

        async def tool_failed(**kwargs):
            raise ReflectToolExecutionError(
                "Reflect tool 'recall' failed on iteration 1: connection to the embedding service was refused"
            )

        monkeypatch.setattr(memory, "reflect_async", tool_failed)
        stub_refresh_has_sources(monkeypatch, memory)

        with pytest.raises(MentalModelRefreshError) as exc_info:
            await memory.refresh_mental_model(
                bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
            )

        # Reported as a typed refresh failure, so the operation records an outcome
        # rather than only prose — and the reflect error stays the cause.
        assert exc_info.value.outcome == "refresh_failed_error"
        assert exc_info.value.reason == "retrieval_failed"
        assert isinstance(exc_info.value.__cause__, ReflectToolExecutionError)

        preserved = await memory.get_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        assert preserved is not None
        assert preserved["content"] == EXISTING, "A failed retrieval overwrote the stored document (#2894)"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_refresh_does_not_advance_the_watermark_on_tool_failure(memory, request_context, monkeypatch):
    """The failed run must stay repeatable.

    ``last_refreshed_at`` bounds the next delta refresh's window. Moving it for a
    run whose retrieval never completed would put the facts this refresh was
    triggered by permanently behind the watermark.
    """
    bank_id = f"test-mm-tool-fail-wm-{uuid.uuid4().hex[:8]}"
    try:
        mm = await _model_with_content(memory, request_context, bank_id)
        before = await memory.get_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        async def tool_failed(**kwargs):
            raise ReflectToolExecutionError("Reflect tool 'search_observations' failed on iteration 2: index missing")

        monkeypatch.setattr(memory, "reflect_async", tool_failed)
        stub_refresh_has_sources(monkeypatch, memory)

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
async def test_refresh_still_writes_when_retrieval_genuinely_found_nothing(memory, request_context, monkeypatch):
    """A working retrieval that returns nothing is an answer, and must be written.

    This is the line the guard draws. The failure mode above is "we could not
    look"; this is "we looked and there is nothing there" — a legitimate result
    that the document is supposed to reflect, so the refresh proceeds normally.
    """
    bank_id = f"test-mm-empty-ok-{uuid.uuid4().hex[:8]}"
    try:
        mm = await _model_with_content(memory, request_context, bank_id)

        answer = "There is no information about the team in this memory bank."

        async def empty_but_successful(**kwargs):
            from hindsight_api.engine.response_models import ReflectResult

            return ReflectResult(text=answer, based_on={})

        monkeypatch.setattr(memory, "reflect_async", empty_but_successful)
        stub_refresh_has_sources(monkeypatch, memory)

        await memory.refresh_mental_model(bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context)

        updated = await memory.get_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        assert updated is not None
        assert answer in updated["content"], "A successful empty retrieval must still update the document"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_failed_refresh_is_recorded_in_the_model_history(memory, request_context, monkeypatch):
    """The failure must be visible on the model itself, not only on the operation.

    A refresh that writes nothing used to leave the model looking as though it had
    simply never been refreshed: the History tab kept rendering the last successful
    trace as current, and the only record was prose on an async-operation row that
    no mental-model view reads (#2894).
    """
    bank_id = f"test-mm-fail-history-{uuid.uuid4().hex[:8]}"
    try:
        mm = await _model_with_content(memory, request_context, bank_id)

        async def tool_failed(**kwargs):
            raise ReflectToolExecutionError("Reflect tool 'recall' failed on iteration 1: pgvector index missing")

        monkeypatch.setattr(memory, "reflect_async", tool_failed)
        stub_refresh_has_sources(monkeypatch, memory)

        with pytest.raises(MentalModelRefreshError):
            await memory.refresh_mental_model(
                bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
            )

        history = await memory.get_mental_model_history(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        assert history, "the failed refresh left no history entry"
        entry = history[0]
        assert entry["kind"] == "refresh_failed"
        assert entry["outcome"] == "refresh_failed_error"
        assert entry["failure_reason"] == "retrieval_failed"
        assert "pgvector index missing" in entry["error_message"]
        # A failure is an event, not a version: there is no content to diff.
        assert entry["previous_content"] is None
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_failure_records_do_not_evict_content_versions(memory, request_context, monkeypatch):
    """Retention is per kind, so a run of failures cannot erase the version history.

    A single overall cap would let a broken retriever push every real version out of
    the table — losing the document's actual history to the noise of the outage that
    stopped it changing.
    """
    bank_id = f"test-mm-fail-retention-{uuid.uuid4().hex[:8]}"
    try:
        mm = await _model_with_content(memory, request_context, bank_id)

        # One successful refresh, so there is a version snapshot to protect.
        async def wrote_content(**kwargs):
            from hindsight_api.engine.response_models import ReflectResult

            return ReflectResult(text="# Team\n\nAlice leads platform.\n", based_on={})

        monkeypatch.setattr(memory, "reflect_async", wrote_content)
        stub_refresh_has_sources(monkeypatch, memory)
        await memory.refresh_mental_model(bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context)
        versions_before = [
            e
            for e in (
                await memory.get_mental_model_history(
                    bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
                )
                or []
            )
            if e.get("kind") != "refresh_failed"
        ]
        assert versions_before, "no version snapshot to protect"

        async def tool_failed(**kwargs):
            raise ReflectToolExecutionError("Reflect tool 'recall' failed on iteration 1: still broken")

        monkeypatch.setattr(memory, "reflect_async", tool_failed)
        stub_refresh_has_sources(monkeypatch, memory)
        for _ in range(12):
            with pytest.raises(MentalModelRefreshError):
                await memory.refresh_mental_model(
                    bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
                )

        history = await memory.get_mental_model_history(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        assert history is not None
        versions_after = [e for e in history if e.get("kind") != "refresh_failed"]
        assert len(versions_after) == len(versions_before), "failure records evicted the version history"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
