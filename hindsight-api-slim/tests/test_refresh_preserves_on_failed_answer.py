"""A refresh that produces no usable answer must keep the previous content.

The reflect agent substitutes a human-readable placeholder when it cannot produce
an answer: ``NO_ANSWER_TEXT`` when the model returns nothing, ``ITERATION_LIMIT_TEXT``
when the agent runs out of iterations. Both are non-empty, so ``refresh_mental_model``'s
"refuse to overwrite with an empty render" guard did not fire for them and the
placeholder was stored over the existing document — while the operation reported
success.

The fix is ``ReflectAgentResult.answer_failure_reason``: set where the failure is
known, checked before any delta rendering, and independent of the placeholder text.
These tests pin that behavior end to end. They patch ``reflect_async`` rather than
``refresh_mental_model`` so the real persistence path — delta, guards, the write —
actually executes; stubbing the refresh itself would assert nothing about storage.
"""

import uuid

import pytest

from hindsight_api import MemoryEngine
from hindsight_api.engine.memory_engine import MentalModelRefreshError, is_populated_content
from hindsight_api.engine.reflect.models import ITERATION_LIMIT_TEXT, NO_ANSWER_TEXT
from hindsight_api.engine.response_models import ReflectResult

HEALTHY_CONTENT = "# Working Document\n\nThis is the real synthesized content that must survive."

# Non-empty evidence, mirroring the production failures: recall returned plenty of
# facts and the answer still came back unusable. Preserving this distinguishes the
# defect from the empty-recall case and keeps grounding for the next refresh.
SUPPORTING_FACTS = [
    {"id": "f1", "text": "supporting fact one", "type": "observation", "context": None},
    {"id": "f2", "text": "supporting fact two", "type": "observation", "context": None},
]


def _reflect_result(text: str, *, failure_reason: str | None, facts: list[dict] | None = None) -> ReflectResult:
    return ReflectResult.model_validate(
        {
            "text": text,
            "answer_failure_reason": failure_reason,
            "based_on": {
                "observation": facts if facts is not None else SUPPORTING_FACTS,
                "world": [],
                "experience": [],
                "mental-models": [],
                "directives": [],
            },
        }
    )


@pytest.fixture
def patch_reflect(monkeypatch):
    """Patch reflect_async only, so refresh_mental_model itself still runs for real."""

    def _install(memory: MemoryEngine, result: ReflectResult):
        async def fake_reflect_async(**kwargs):
            return result

        monkeypatch.setattr(memory, "reflect_async", fake_reflect_async)

    return _install


@pytest.fixture
async def bank_with_model(memory: MemoryEngine, request_context):
    """Bank holding one mental model with known-good content; unique per test for xdist."""
    bank_id = f"test-refresh-preserve-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id, request_context=request_context)
    mm = await memory.create_mental_model(
        bank_id=bank_id,
        name="Preservation Model",
        source_query="What must survive a failed refresh?",
        content=HEALTHY_CONTENT,
        request_context=request_context,
    )
    yield memory, bank_id, mm
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_empty_answer_preserves_content_and_raises(bank_with_model, request_context, patch_reflect):
    """The no-answer placeholder must not replace a working document."""
    memory, bank_id, mm = bank_with_model
    patch_reflect(memory, _reflect_result(NO_ANSWER_TEXT, failure_reason="empty_answer"))

    with pytest.raises(MentalModelRefreshError):
        await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

    after = await memory.get_mental_model(bank_id, mm["id"], request_context=request_context)
    assert after["content"] == HEALTHY_CONTENT
    assert after["reflect_response"]["refresh_skipped"] == "empty_answer"


@pytest.mark.asyncio
async def test_iteration_limit_preserves_content_and_raises(bank_with_model, request_context, patch_reflect):
    """The iteration-limit placeholder is a different failure and must also be caught.

    A guard written against the no-answer text alone would let this one through.
    """
    memory, bank_id, mm = bank_with_model
    patch_reflect(memory, _reflect_result(ITERATION_LIMIT_TEXT, failure_reason="iteration_limit"))

    with pytest.raises(MentalModelRefreshError):
        await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

    after = await memory.get_mental_model(bank_id, mm["id"], request_context=request_context)
    assert after["content"] == HEALTHY_CONTENT
    assert after["reflect_response"]["refresh_skipped"] == "iteration_limit"


@pytest.mark.asyncio
async def test_failed_refresh_preserves_supporting_evidence(bank_with_model, request_context, patch_reflect):
    """The failure record keeps this refresh's based_on rather than blanking it.

    Two reasons: it is the evidence that the answer failed *despite* real retrieval,
    and delta refreshes accumulate grounding from the stored reflect_response, so
    erasing it would degrade the next successful refresh.
    """
    memory, bank_id, mm = bank_with_model
    patch_reflect(memory, _reflect_result(NO_ANSWER_TEXT, failure_reason="empty_answer"))

    with pytest.raises(MentalModelRefreshError):
        await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

    after = await memory.get_mental_model(bank_id, mm["id"], request_context=request_context)
    stored_ids = [f["id"] for f in after["reflect_response"]["based_on"]["observation"]]
    assert stored_ids == ["f1", "f2"]


@pytest.mark.asyncio
async def test_delta_mode_cannot_launder_a_failed_answer(bank_with_model, request_context, patch_reflect):
    """Delta rendering must never get the chance to turn a placeholder into content.

    Delta operates on the candidate and re-renders the document, so a check placed
    after it could see ordinary-looking markdown. The failure check therefore runs
    before delta; this test drives the same refresh twice, since delta mode only
    engages once a prior refresh has recorded a matching source query.
    """
    memory, bank_id, mm = bank_with_model

    patch_reflect(memory, _reflect_result("# Real Synthesis\n\nEstablished baseline.", failure_reason=None))
    await memory.refresh_mental_model(bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context)
    baseline = await memory.get_mental_model(bank_id, mm["id"], request_context=request_context)
    assert baseline["content"].startswith("# Real Synthesis")

    patch_reflect(memory, _reflect_result(NO_ANSWER_TEXT, failure_reason="empty_answer"))
    with pytest.raises(MentalModelRefreshError):
        await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

    after = await memory.get_mental_model(bank_id, mm["id"], request_context=request_context)
    assert after["content"] == baseline["content"]


@pytest.mark.asyncio
async def test_successful_refresh_still_writes_content(bank_with_model, request_context, patch_reflect):
    """The guard must not block real answers — the failure path is the exception."""
    memory, bank_id, mm = bank_with_model
    new_content = "# Updated Synthesis\n\nFresh content from a successful reflect."
    patch_reflect(memory, _reflect_result(new_content, failure_reason=None))

    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
    )

    assert refreshed["content"] == new_content
    assert "refresh_skipped" not in refreshed["reflect_response"]
    after = await memory.get_mental_model(bank_id, mm["id"], request_context=request_context)
    assert after["content"] == new_content


@pytest.mark.asyncio
async def test_async_refresh_of_failed_answer_does_not_report_success(
    bank_with_model, request_context, patch_reflect
):
    """Through the task path, a failed refresh never settles as a successful one.

    This is the property the incident violated: the operation reported success while
    the stored document had been replaced by a placeholder. The failure must surface
    as an error and the content must be untouched.

    MentalModelRefreshError is classified retryable, so the task layer re-raises it as
    RetryTaskAt: a transient provider hiccup gets another attempt, and only exhausted
    retries settle as failed. Preservation is what makes that safe to retry — each
    attempt leaves the existing document intact. The test backend runs the task inline
    on submit, so that wrapper surfaces here.
    """
    from hindsight_api.worker.exceptions import RetryTaskAt

    memory, bank_id, mm = bank_with_model
    patch_reflect(memory, _reflect_result(NO_ANSWER_TEXT, failure_reason="empty_answer"))

    with pytest.raises(RetryTaskAt):
        await memory.submit_async_refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

    after = await memory.get_mental_model(bank_id, mm["id"], request_context=request_context)
    assert after["content"] == HEALTHY_CONTENT
    assert after["reflect_response"]["refresh_skipped"] == "empty_answer"


def test_is_populated_content_rejects_every_placeholder():
    """The text-level backstop knows all three placeholders, not just the empty string."""
    assert is_populated_content("# Real content") is True
    assert is_populated_content("") is False
    assert is_populated_content("   \n  ") is False
    assert is_populated_content(None) is False
    assert is_populated_content(NO_ANSWER_TEXT) is False
    assert is_populated_content(ITERATION_LIMIT_TEXT) is False
    assert is_populated_content("Generating content...") is False
