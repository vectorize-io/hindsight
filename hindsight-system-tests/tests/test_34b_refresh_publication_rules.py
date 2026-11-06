"""A deployment can refuse a refreshed page before readers see it.

Some pages have publication rules the model cannot be trusted to follow: a hard length, no
repeated sections, a word that must never appear. The operation validator sees each finished
page before it is stored (#5970). The server's validator (`server_extensions/publication_rules_validator.py`)
refuses any page that says UNPUBLISHABLE.

Two contracts meet here. A refused page leaves the last accepted one in place, and the refresh
fails for good — no worker retry regenerates the same page unasked. A page whose validator asks
for another draft gets one in the same run, and the redraft is told why the first was refused:
without the reason the model writes the same page again.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests import reflect_loop
from hindsight_system_tests.payloads import consolidation, extracted, fact
from hindsight_system_tests.waiting import wait_until_settled

pytestmark = pytest.mark.asyncio

ACCEPTED = "Alice lives in Berlin."
REFUSED = "Alice lives in Berlin. UNPUBLISHABLE"
# Mirrors REASON in the server's validator.
REASON = "the page must not contain the word UNPUBLISHABLE"


async def _bank_with_a_fact(admin, llm, bank_id: str) -> None:
    llm.on_step("extract_facts").returns(extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice"])))
    llm.on_step("consolidate").returns(consolidation())
    await admin.aretain(bank_id=bank_id, content="Alice moved to Berlin")
    await wait_until_settled(admin, bank_id)


async def _create(admin, bank_id: str, name: str) -> str:
    created = await admin.mental_models.create_mental_model(
        bank_id, {"name": name, "source_query": "Where does Alice live?"}
    )
    return created.mental_model_id


async def _content(admin, bank_id: str, model: str) -> str:
    return (await admin.mental_models.get_mental_model(bank_id, model, detail="full")).content.strip()


async def test_a_refused_page_keeps_the_last_accepted_one(publishing_client, llm, publishing_bank):
    admin, bank_id = publishing_client, publishing_bank
    await _bank_with_a_fact(admin, llm, bank_id)
    reflect_loop(llm, answer=ACCEPTED)
    model = await _create(admin, bank_id, "Housing")
    await wait_until_settled(admin, bank_id)
    assert await _content(admin, bank_id, model) == ACCEPTED

    llm.reset()
    reflect_loop(llm, answer=REFUSED)
    await admin.mental_models.refresh_mental_model(bank_id, model)
    await wait_until_settled(admin, bank_id, allow_failed=True)

    # Readers still get the accepted page.
    assert await _content(admin, bank_id, model) == ACCEPTED
    # The refresh failed for good, with the validator's reason on record. A worker retry
    # would have left it pending.
    failed = await admin.operations.list_operations(bank_id, type="refresh_mental_model", status="failed")
    assert failed.total == 1
    status = await admin.operations.get_operation_status(bank_id, failed.operations[0].id)
    assert status.details.outcome == "refresh_failed_candidate_rejected"
    assert status.details.failure_reason == "candidate_rejected"
    history = await admin.mental_models.get_mental_model_history(bank_id, model)
    assert history[0]["kind"] == "refresh_failed"
    assert history[0]["failure_reason"] == "candidate_rejected"
    assert REASON in history[0]["error_message"]


async def test_a_redraft_is_told_why_the_first_draft_was_refused(publishing_client, llm, publishing_bank):
    admin, bank_id = publishing_client, publishing_bank
    await _bank_with_a_fact(admin, llm, bank_id)
    # The model only writes an acceptable page once its prompt says what was wrong.
    # Rules match in registration order, so the redraft's whole loop (as in `reflect_loop`)
    # goes before the first draft's.
    llm.on_step("reflect", contains=REASON, tool="done").returns_tool_call("done", answer=ACCEPTED)
    llm.on_step("reflect", contains=REASON, tool="answer_ready").returns_tool_call("answer_ready", answer_found=True)
    llm.on_step("reflect", contains=REASON).calls_the_offered_tool(query="system test")
    llm.on_step("reflect", contains=REASON).returns_text(ACCEPTED)
    reflect_loop(llm, answer=REFUSED)

    model = await _create(admin, bank_id, "Redraft housing")
    await wait_until_settled(admin, bank_id)

    assert await _content(admin, bank_id, model) == ACCEPTED
    operations = await admin.operations.list_operations(bank_id, type="refresh_mental_model")
    assert [op.status for op in operations.operations] == ["completed"]
    status = await admin.operations.get_operation_status(bank_id, operations.operations[0].id)
    assert status.details.outcome == "content_written"
