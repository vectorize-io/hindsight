"""An operator retunes LLM concurrency while memory work is running.

The cap is lowered to 1 before a retain and restored after it. A lowered cap only
queues LLM calls — it must never fail them — so the retain still completes and its
facts are still recalled. DELETE puts back the configured value, and by then no
call is left holding a permit.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio


async def test_work_retained_under_a_lowered_cap_still_completes(client, llm, bank_id, settled):
    configured = (await client.monitoring.get_llm_concurrency()).configured_max_concurrent
    try:
        lowered = await client.monitoring.update_llm_concurrency({"max_concurrent": 1})
        assert lowered.max_concurrent == 1

        llm.on_step("extract_facts", contains="Oslo").returns(
            extracted(fact("Bea moved to Oslo in 2024", when="2024", where="Oslo", who="Bea", entities=["Bea", "Oslo"]))
        )
        llm.on_step("consolidate").returns(consolidation())
        await client.aretain(bank_id=bank_id, content="Bea moved to Oslo in 2024.")
        await settled(bank_id)
    finally:
        # The server is shared by every story: never leave it capped at 1.
        restored = await client.monitoring.reset_llm_concurrency()
    assert restored.max_concurrent == configured
    assert restored.in_flight == 0

    recalled = await client.arecall(bank_id=bank_id, query="Where did Bea move?")
    assert [result.text for result in recalled.results] == ["Bea moved to Oslo in 2024 | When: 2024 | Involving: Bea"]
