"""#5938: an amendment retained first is not overwritten by the older rule it amends.

Two sessions are imported together: 10:48 amends the overage rule (the first 25 units are free),
09:48 is the rule it amends (no free units), worded as "was updated". The amendment lands first,
then the older rule arrives. Consolidation must keep the grace allowance as the current rule —
the older fact may only add history. gemini-3.1-flash-lite got this wrong every time before the
prompt's TIME section and the reason-first, required-field UPDATE schema.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

import pytest

from hindsight_api.config import _get_raw_config
from hindsight_api.engine.memory_engine import MemoryEngine
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core

AMENDMENT = (
    "Amended overage policy: overage now includes a 25-unit monthly grace allowance applied before "
    "block rounding: billable = max(0, overage - 25), then rounded up into 50-unit blocks."
)
OLDER_RULE = (
    "The metering policy for overage_charge was updated to use prorated quotas and block-based billing: "
    "overage bills in 50-unit blocks rounded up. No waivers for small overages are permitted."
)


@pytest.fixture(autouse=True)
def enable_observations():
    config = _get_raw_config()
    original = config.enable_observations
    config.enable_observations = True
    yield
    config.enable_observations = original


@pytest.mark.asyncio
async def test_older_rule_arriving_second_does_not_overwrite_its_amendment(
    memory_real_llm: MemoryEngine, request_context: Any
) -> None:
    memory = memory_real_llm
    bank_id = f"test-5938-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)

    async def retain(content: str, event_date: datetime, document_id: str) -> None:
        await memory.retain_async(
            bank_id=bank_id,
            content=content,
            event_date=event_date,
            document_id=document_id,
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()

    await retain(AMENDMENT, datetime(2026, 10, 21, 10, 48, tzinfo=timezone.utc), "session-02")
    await retain(OLDER_RULE, datetime(2026, 10, 21, 9, 48, tzinfo=timezone.utc), "session-01")

    units = (await memory.list_memory_units(bank_id, limit=1000, request_context=request_context))["items"]
    print("\n" + "\n".join(f"- [{u['fact_type']}] (mentioned_at={u['mentioned_at']}) {u['text']}" for u in units))

    observations = "\n".join(f"- {u['text']}" for u in units if u["fact_type"] == "observation")
    await assert_meets_criteria(
        response=observations,
        criteria=(
            "The observations state that the first 25 overage units each month are free (a grace allowance "
            "applied before block rounding) as the CURRENT rule. Mentioning the no-waiver block billing as an "
            "earlier rule is fine; saying the grace allowance was removed or replaced, or that no free overage "
            "units are allowed now, fails."
        ),
        context=(
            "The no-waiver rule was stated at 09:48 and the 25-unit grace allowance amended it at 10:48 the same "
            "day. The amendment was retained first, the older rule second."
        ),
    )

    await memory.delete_bank(bank_id, request_context=request_context)
