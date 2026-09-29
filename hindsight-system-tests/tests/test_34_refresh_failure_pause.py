"""A mental model whose refresh failed is not refreshed again automatically.

A failed refresh leaves the model stale, and staleness is what the automatic
triggers look for. So a refresh that could never succeed — a prompt the model
cannot answer in time, an account out of credit — was queued again after every
consolidation and every scheduler tick, and paid the LLM for the same failure
each time. One install spent its whole balance overnight while idle (#4532).

The contract: once a refresh fails, the automatic triggers leave that model
alone. A refresh someone asks for still runs, and when it succeeds the automatic
ones resume.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests import reflect_loop, wait_until_settled
from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

ANSWER = "Alice lives in Berlin."


async def _refreshes(client, bank_id: str, status: str | None = None) -> int:
    operations = await client.operations.list_operations(
        bank_id, type="refresh_mental_model", status=status, limit=100
    )
    return operations.total


async def _retain(client, llm, bank_id: str, text: str) -> None:
    llm.on_step("extract_facts").returns(extracted(fact(text, who="Alice", entities=["Alice"])))
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content=text)
    # The failed refresh is this story's subject; it asserts that operation itself.
    await wait_until_settled(client, bank_id, allow_failed=True)


async def test_a_failed_refresh_is_not_retried_by_every_consolidation(client, llm, bank_id, settled):
    reflect_loop(llm, answer=ANSWER)
    await _retain(client, llm, bank_id, "Alice moved to Berlin")
    created = await client.mental_models.create_mental_model(
        bank_id,
        {
            "name": "Housing",
            "source_query": "Where does Alice live?",
            "trigger": {"refresh_after_consolidation": True},
        },
    )
    await settled(bank_id)
    model = created.mental_model_id
    baseline = await _refreshes(client, bank_id)

    # The next consolidation triggers a refresh, and that refresh fails: the agent
    # ends without an answer.
    llm.reset()
    reflect_loop(llm, answer="")
    await _retain(client, llm, bank_id, "Alice bought a bike")
    assert await _refreshes(client, bank_id) == baseline + 1
    assert await _refreshes(client, bank_id, status="failed") == 1
    history = await client.mental_models.get_mental_model_history(bank_id, model)
    assert history[0]["kind"] == "refresh_failed"

    # Two more consolidations. Before the fix each one queued the same doomed refresh.
    await _retain(client, llm, bank_id, "Alice started a new job")
    await _retain(client, llm, bank_id, "Alice adopted a cat")
    assert await _refreshes(client, bank_id) == baseline + 1

    # A refresh someone asks for still runs; its success resumes the automatic ones.
    llm.reset()
    reflect_loop(llm, answer=ANSWER)
    await client.mental_models.refresh_mental_model(bank_id, model)
    await wait_until_settled(client, bank_id, allow_failed=True)
    assert (await client.mental_models.get_mental_model(bank_id, model, detail="full")).content.strip() == ANSWER

    await _retain(client, llm, bank_id, "Alice visited Paris")
    assert await _refreshes(client, bank_id) == baseline + 3
