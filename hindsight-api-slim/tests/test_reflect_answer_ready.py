"""A real model says "enough" with answer_ready and the closing step writes the answer once (#5832).

Before, the free turn that ended retrieval wrote the whole answer (as prose, or as a ``done`` the
agent mode throws away) and the closing step wrote it again: about half of a reflect's output
tokens were never shown to anyone. The mechanics are pinned with scripted providers in
test_reflect_agent.py (TestFreeTurnsSignalReady); this checks that a real model follows the
prompt and that the answer it then writes still holds everything the evidence said.
"""

from unittest.mock import AsyncMock

import pytest

from hindsight_api.engine.reflect.agent import run_reflect_agent
from tests.llm_judge import assert_meets_criteria

pytestmark = [pytest.mark.hs_llm_core, pytest.mark.flaky(reruns=3, reruns_delay=2)]

_FACTS = [
    "The billing service moved from Stripe to Adyen in March 2026 because Stripe fees on EU cards were 2.9%.",
    "Adyen fees for EU cards are 1.4% plus 0.10 EUR per transaction.",
    "Invoices are generated nightly at 02:00 UTC by the invoice-cron job.",
    "Customers on the Enterprise plan are invoiced quarterly; everyone else monthly.",
    "Failed payments trigger a dunning email after 1, 3 and 7 days.",
    "After 14 days of failed payment the account is suspended, not deleted.",
    "The suspension rule was decided in May 2026 after a customer lost data when accounts were deleted.",
]

_QUERY = (
    "How does billing work today: payment provider and why, fees, invoicing schedule, "
    "and what happens on failed payments and why?"
)


@pytest.mark.asyncio
async def test_the_answer_is_written_once_and_keeps_every_fact(llm_config):
    free_turns: list[list[str]] = []
    original = llm_config.call_with_tools

    async def recording(**kwargs):
        result = await original(**kwargs)
        if kwargs.get("tool_choice") is not None and kwargs["tool_choice"].function_name is None:
            free_turns.append([tc.name for tc in result.tool_calls] or ["<prose>"])
        return result

    llm_config.call_with_tools = recording
    try:
        result = await run_reflect_agent(
            llm_config=llm_config,
            bank_id="answer-ready",
            query=_QUERY,
            bank_profile={"name": "Billing", "mission": "Answer questions about the company's billing."},
            search_mental_models_fn=AsyncMock(return_value={"mental_models": []}),
            read_mental_models_fn=AsyncMock(return_value={"mental_models": []}),
            search_observations_fn=AsyncMock(return_value={"observations": []}),
            recall_fn=AsyncMock(
                return_value={"memories": [{"id": f"mem-{i}", "text": t} for i, t in enumerate(_FACTS)]}
            ),
            expand_fn=AsyncMock(return_value={"memories": []}),
            has_mental_models=False,
            budget="mid",
            max_iterations=10,
        )
    finally:
        llm_config.call_with_tools = original

    scopes = [c.scope for c in result.llm_trace]
    assert scopes[-1] == "closing_done", scopes
    assert free_turns and free_turns[-1] == ["answer_ready"], (
        f"the turn that ended retrieval should only signal it, not write the answer: {free_turns}"
    )
    await assert_meets_criteria(
        response=result.text,
        criteria=(
            "The answer says billing moved to Adyen because Stripe's EU card fees were 2.9%, gives Adyen's fee "
            "(1.4% + 0.10 EUR), the invoicing schedule (nightly at 02:00 UTC; Enterprise quarterly, others "
            "monthly), and that failed payments get dunning emails after 1, 3 and 7 days and the account is "
            "suspended (not deleted) after 14 days because a customer once lost data to deletion."
        ),
        context="The evidence was exactly these facts: " + " ".join(_FACTS),
    )
