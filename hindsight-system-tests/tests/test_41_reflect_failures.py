"""Reflect fails loudly rather than returning a plausible non-answer.

Reflect is agentic: it needs a model that calls tools, and it needs that model to
eventually produce an answer. Both can fail — a provider without function
calling, a model that returns empty content, output truncated before the answer
was written.

The tempting behaviour is to degrade: return an empty string, or a placeholder,
or whatever prose happened to arrive instead of a tool call. That is the worst
option available, because the caller is an agent that will treat the answer as
memory and act on it. A confident empty answer is indistinguishable from "your
bank has nothing about this", and the two demand opposite responses.

So each of these must raise, and the message must name the cause — someone
pointing reflect at a model that cannot call tools needs to be told that, not
handed a blank.
"""

from __future__ import annotations

import pytest
from hindsight_client_api.exceptions import ServiceException

from hindsight_system_tests import reflect_loop
from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

QUERY = "Where does Alice live?"


@pytest.fixture
async def bank_with_facts(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.")
    await settled(bank_id)
    return bank_id


async def test_a_model_that_never_calls_a_tool_is_an_error(client, llm, bank_with_facts):
    """The provider-mismatch case: reflect's first turn forces a tool, and this
    model answers with prose instead.

    The failure has to name the cause. "No answer" sends someone looking at their
    data; "this model produced no usable tool call" sends them at their config.
    """
    llm.on_step("reflect").returns_text("I think Alice lives in Berlin.")

    with pytest.raises(ServiceException) as raised:
        await client.areflect(bank_id=bank_with_facts, query=QUERY)

    assert "tool" in str(raised.value).lower()


async def test_an_empty_answer_is_an_error_rather_than_an_empty_result(client, llm, bank_with_facts):
    """A model that climbs the ladder and then says nothing.

    Returning "" here would be the dangerous degradation: an agent cannot tell it
    apart from a bank that genuinely holds nothing, and those call for opposite
    behaviour — one is "go and find out", the other is "you already know".
    """
    llm.on_step("reflect", tool="done").returns_tool_call("done", answer="")
    llm.on_step("reflect").calls_the_offered_tool(query="Alice")
    llm.on_step("reflect").returns_text("")

    with pytest.raises(ServiceException):
        await client.areflect(bank_id=bank_with_facts, query=QUERY)


async def test_a_working_model_still_answers(client, llm, bank_with_facts):
    """The control. Without it, the two tests above would also pass if reflect
    were broken outright."""
    reflect_loop(llm, answer="Alice lives in Berlin.")

    response = await client.areflect(bank_id=bank_with_facts, query=QUERY)

    assert response.text == "Alice lives in Berlin."
