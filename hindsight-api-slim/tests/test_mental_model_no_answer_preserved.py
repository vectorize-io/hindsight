"""A reflect run that produced no answer must not touch the stored document (#2959).

Reflect used to hand back the placeholder "No answer provided." whenever the
``done`` tool arrived with a blank answer. The placeholder is non-empty, so
``refresh_mental_model``'s "refuse to overwrite existing content with an empty
render" guard (``not final_content.strip()``) let it through: a mental model
built across months of refreshes was replaced by a 19-character string, and the
async operation was recorded as ``completed`` with ``error_message: null`` —
no failure signal anywhere to alert on.

Reflect now raises instead of inventing text, so the refresh never reaches its
write. These tests assert that end of it: the exception propagates and the
stored document is byte-identical afterwards.
"""

import uuid

import pytest

from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.reflect import ReflectNoAnswerError
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
async def test_refresh_preserves_content_when_reflect_has_no_answer(memory, request_context, monkeypatch):
    """The reported production case: evidence was gathered, the answer never arrived."""
    bank_id = f"test-mm-no-answer-{uuid.uuid4().hex[:8]}"
    try:
        mm = await _model_with_content(memory, request_context, bank_id)

        async def no_answer(**kwargs):
            raise ReflectNoAnswerError("Reflect's done tool returned no answer (iteration 3, 5 tool call(s) made).")

        monkeypatch.setattr(memory, "reflect_async", no_answer)
        stub_refresh_has_sources(monkeypatch, memory)

        with pytest.raises(ReflectNoAnswerError):
            await memory.refresh_mental_model(
                bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
            )

        preserved = await memory.get_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        assert preserved is not None
        assert preserved["content"] == EXISTING, "A failed reflect overwrote the stored document (#2959)"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_refresh_does_not_advance_the_watermark_on_no_answer(memory, request_context, monkeypatch):
    """The failed run must stay repeatable: nothing it never read may fall out of the window.

    ``last_refreshed_at`` bounds the next delta refresh's window. Moving it for a
    run that produced nothing would put the facts this refresh was triggered by
    permanently behind the watermark, so no later refresh would ever see them.
    """
    bank_id = f"test-mm-no-answer-wm-{uuid.uuid4().hex[:8]}"
    try:
        mm = await _model_with_content(memory, request_context, bank_id)
        before = await memory.get_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        async def no_answer(**kwargs):
            raise ReflectNoAnswerError("Reflect's final synthesis returned no text after 4 iteration(s).")

        monkeypatch.setattr(memory, "reflect_async", no_answer)
        stub_refresh_has_sources(monkeypatch, memory)

        with pytest.raises(ReflectNoAnswerError):
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
