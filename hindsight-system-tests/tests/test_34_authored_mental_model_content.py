"""A mental model can be born with content the caller wrote.

A mental model is normally generated: you give it a question and reflect writes
the answer. That is the wrong shape for knowledge the bank has no way to derive
— a team's conventions, a policy that lives in a wiki, the output of a review
that happened outside Hindsight. The caller has the text; asking the model to
reinvent it is how you get a plausible paraphrase of the one document that had
to be exact.

So the API accepts authored content, stores it as-is, and schedules nothing: no
placeholder, no background reflect, no `operation_id`. The two things a caller
depends on are that the text they handed over is what comes back, and that
creation queued no work to overwrite it. The second is the one that fails
silently — a queued refresh would replace the authored document a moment after
the response, and the caller would only find out by reading it again.

Authored content is not permanent, and these stories do not pretend otherwise:
the next refresh rewrites it according to `trigger.mode` (`full` discards it,
`delta` edits it in place). What is pinned here is the create/update contract,
not a promise that Hindsight will never touch the document again.
"""

from __future__ import annotations

import pytest

from hindsight_client import Hindsight
from hindsight_client_api.models.create_mental_model_response import CreateMentalModelResponse

pytestmark = pytest.mark.asyncio

AUTHORED = "## Deployment conventions\n\n- Roll out behind a feature flag\n- Never page on-call for a single failed retry\n"
SOURCE_QUERY = "What are the team's deployment conventions?"


async def _authored_model(client: Hindsight, bank: str) -> CreateMentalModelResponse:
    """Create the model under test and hand back the create response."""
    return await client.acreate_mental_model(
        bank_id=bank,
        name="Deployment conventions",
        source_query=SOURCE_QUERY,
        content=AUTHORED,
    )


async def test_created_content_is_what_the_caller_wrote(client, bank_id):
    created = await _authored_model(client, bank_id)

    model = await client.mental_models.get_mental_model(bank_id, created.mental_model_id, detail="full")

    assert model.content.strip() == AUTHORED.strip(), "the authored document must come back verbatim"
    assert model.source_query == SOURCE_QUERY, "source_query is still required and still recorded"


async def test_authored_creation_reports_no_operation(client, bank_id):
    """Nothing was queued, so there is nothing to poll.

    `operation_id=None` is the contract; the operations list is the evidence — a
    response that merely omitted the id while still enqueueing a refresh would
    pass the first assertion and fail this one.
    """
    created = await _authored_model(client, bank_id)

    assert created.operation_id is None

    operations = await client.operations.list_operations(bank_id, limit=100)
    assert [op.task_type for op in operations.operations] == [], (
        "authored content must not schedule a create-time refresh"
    )


async def test_authored_creation_never_runs_reflect(client, bank_id, llm):
    """No LLM turn may happen at create time.

    The rulebook is empty, so any reflect call is unscripted and the suite's
    autouse guard fails the test on it. This asserts the stronger property
    directly: creation is a write, not a generation.
    """
    await _authored_model(client, bank_id)

    assert llm.unmatched == []


async def test_update_replaces_authored_content_directly(client, bank_id):
    """Editing the document is a write, not a refresh.

    Same rulebook-empty argument as above: if `PATCH` ran reflect, the call would
    be unscripted and fail. A caller correcting a typo must not pay for an LLM
    round trip, and must not have the rest of the document rewritten around it.
    """
    created = await _authored_model(client, bank_id)
    corrected = AUTHORED + "- Escalate a rollback to the incident channel\n"

    updated = await client.aupdate_mental_model(bank_id, created.mental_model_id, content=corrected)

    assert updated.content.strip() == corrected.strip()

    model = await client.mental_models.get_mental_model(bank_id, created.mental_model_id, detail="full")
    assert model.content.strip() == corrected.strip()
    assert model.source_query == SOURCE_QUERY, "content edits leave the query alone"


async def test_a_content_edit_does_not_require_a_source_query(client, bank_id):
    """`source_query` is required to create, optional to update.

    A caller who only wants to correct the text should not have to restate the
    question the document answers.
    """
    created = await _authored_model(client, bank_id)

    updated = await client.aupdate_mental_model(
        bank_id, created.mental_model_id, content="## Deployment conventions\n\n- Ship small\n"
    )

    assert "Ship small" in updated.content
    assert updated.source_query == SOURCE_QUERY
