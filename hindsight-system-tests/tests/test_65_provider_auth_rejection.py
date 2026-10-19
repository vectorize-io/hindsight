"""A retain whose LLM key is rejected fails once, instead of replaying the rejection.

A 401/403 is a property of the configured key, not of the moment: every replay
sends the same key and earns the same answer. The worker used to treat it as a
passing fault and re-ran the whole retain on its retry schedule, every chunk
again — one install logged 168 identical rejected calls for a single PDF (#5644).

The contract: a rejected key fails the operation on the first attempt, with the
reason in its error. A provider that is down (a 5xx) is still retried, because
that one can come back.
"""

from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.asyncio

#: 1 attempt + the 2 retries ``worker_retry_server`` allows.
ATTEMPTS_WITH_RETRIES = 3


async def _failed_retain(client, llm, bank: str):
    """Submit a retain and wait for it to reach ``failed``; return the operation."""
    await client.aretain(bank_id=bank, content="Alice moved to Berlin.", retain_async=True)
    for _ in range(60):
        listing = await client.operations.list_operations(bank, status="failed", limit=10)
        if listing.operations:
            return listing.operations[0]
        await asyncio.sleep(0.5)
    raise AssertionError(f"the retain never failed; extraction was asked {len(llm.prompts_for('extract_facts'))}x")


@pytest.mark.parametrize("status", [401, 403])
async def test_a_rejected_key_fails_the_retain_on_the_first_attempt(worker_retry_client, llm, worker_retry_bank, status):
    llm.on_step("extract_facts").fails_with_status(status)

    failed = await _failed_retain(worker_retry_client, llm, worker_retry_bank)

    assert len(llm.prompts_for("extract_facts")) == 1, "a rejected key was sent again"
    assert str(status) in (failed.error_message or "")


async def test_a_provider_outage_is_still_retried(worker_retry_client, llm, worker_retry_bank):
    """The control: without it the story above would pass on a server that never retries."""
    llm.on_step("extract_facts").fails_with_status(503)

    await _failed_retain(worker_retry_client, llm, worker_retry_bank)

    assert len(llm.prompts_for("extract_facts")) == ATTEMPTS_WITH_RETRIES
