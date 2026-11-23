"""A retain the provider's content filter blocks fails once, and says why (#6503).

A block is a property of the text, not of the moment: every retry gets the same
answer. So it must not run the retry schedule the way a provider blip does. The
OpenAI-compatible providers used to report it as an ordinary empty reply, which
the worker replayed through every retry and HTTP answered as a 500.
"""

from __future__ import annotations

import asyncio

import pytest
from hindsight_client_api.exceptions import ApiException

pytestmark = pytest.mark.asyncio

BLOCKED = "Alice wrote something the provider will not read."


async def test_a_blocked_retain_answers_422(client, llm, bank_id):
    await client.acreate_bank(bank_id=bank_id, name="Alice")
    llm.on_step("extract_facts", contains="will not read").is_blocked()

    with pytest.raises(ApiException) as refused:
        await client.aretain(bank_id=bank_id, content=BLOCKED)

    assert refused.value.status == 422
    assert len(llm.prompts_for("extract_facts")) == 1


async def test_a_blocked_async_retain_fails_without_a_retry(worker_retry_client, llm, worker_retry_bank):
    client, bank = worker_retry_client, worker_retry_bank
    await client.acreate_bank(bank_id=bank, name="Alice")
    llm.on_step("extract_facts", contains="will not read").is_blocked()

    await client.aretain(bank_id=bank, content=BLOCKED, retain_async=True)

    # This server's worker retries a failed operation twice, a second apart; a blip
    # would be pending again here, not failed.
    for _ in range(60):
        failed = await client.operations.list_operations(bank, status="failed", limit=100)
        if any(op.task_type == "retain" for op in failed.operations):
            break
        await asyncio.sleep(1)
    else:
        raise AssertionError("the blocked retain never reported its failure")

    # The batch that carried it fails with it; the retain itself names the refusal.
    retain = next(op for op in failed.operations if op.task_type == "retain")
    assert retain.error_message.startswith("ProviderContentPolicyError:")
    assert len(llm.prompts_for("extract_facts")) == 1
