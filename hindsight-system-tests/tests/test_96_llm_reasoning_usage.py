"""Reasoning tokens survive a real retain and appear in LLM request history.

The OpenAI-compatible stub reports 20 visible completion tokens and 60 hidden
reasoning tokens for fact extraction. The API must preserve those as separate
quantities, both for the individual request and for the bank's chart totals.
"""

from __future__ import annotations

import asyncio

import pytest
from hindsight_client_api.api.llm_traces_api import LLMTracesApi

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio


async def test_reasoning_usage_is_visible_in_llm_request_history(client, llm, bank_id, settled):
    llm.on_step("extract_facts", contains="reasoning usage story").returns(
        extracted(fact("A reasoning usage story was retained")),
        visible_tokens=20,
        reasoning_tokens=60,
    )
    llm.on_step("consolidate").returns(consolidation())

    await client.aretain(bank_id=bank_id, content="This is a reasoning usage story.")
    await settled(bank_id)

    traces = LLMTracesApi(client._api_client)
    # Trace writes are fire-and-forget. Poll their public read endpoint rather
    # than assuming the operations queue settling also flushed the recorder.
    for _ in range(50):
        listing = await traces.list_llm_requests(bank_id, operation="retain", scope="retain_extract_facts")
        if listing.items:
            break
        await asyncio.sleep(0.1)
    else:
        raise AssertionError("fact extraction produced no LLM request trace")

    assert len(listing.items) == 1
    request = listing.items[0]
    assert request.status == "success"
    assert request.output_tokens == 20
    assert request.thoughts_tokens == 60
    assert request.total_tokens == request.input_tokens + 20

    stats = await traces.llm_request_stats(bank_id, operation="retain", period="1d")
    assert sum(bucket.tokens.output for bucket in stats.buckets) == 20
    assert sum(bucket.tokens.thoughts for bucket in stats.buckets) == 60
