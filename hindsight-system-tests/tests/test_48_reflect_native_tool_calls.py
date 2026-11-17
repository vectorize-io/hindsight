"""Reflect works with a model that answers in native tool calls (#6520).

Reflect sends no tools: it describes them in the prompt, asks for one by name and holds
the reply to a response schema. gpt-oss (on Bedrock and on Ollama) answers that with a
native tool call and empty text, on every turn. Read as text only, that was "no usable
call" on the first step, and every reflect failed.

It also mixes up the closing ``done`` with ``answer_ready``: it sends a done holding only
``answer_found``. That one is sent back once, saying the answer is missing. Empty again,
reflect still fails loudly, never with an empty answer.
"""

from __future__ import annotations

import pytest
from hindsight_client_api.exceptions import ServiceException

from hindsight_system_tests.payloads import consolidation, extracted, fact
from hindsight_system_tests.rulebook import ChatRequest, ToolCall

pytestmark = pytest.mark.asyncio

QUERY = "Where does Alice live?"
ANSWER = "Alice lives in Berlin."


@pytest.fixture
async def bank_with_facts(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.")
    await settled(bank_id)
    return bank_id


def _native_model(llm, done) -> None:
    """Every tool turn answered with a native call, the way gpt-oss answers."""
    llm.on_step("reflect", tool="done").answers_with_tool_call(done, native=True)
    llm.on_step("reflect", tool="answer_ready").returns_tool_call("answer_ready", native=True, answer_found=True)
    llm.on_step("reflect").calls_the_offered_tool(native=True, query="Alice")
    llm.on_step("reflect").returns_text(ANSWER)


async def test_a_model_that_calls_tools_natively_still_answers(client, llm, bank_with_facts):
    def done(request: ChatRequest) -> ToolCall:
        # The first done holds only answer_found; told the answer is missing, it writes it.
        if request.has_result_of("done"):
            return ToolCall("done", {"answer": ANSWER, "answer_found": True})
        return ToolCall("done", {"answer_found": True})

    _native_model(llm, done)

    result = await client.areflect(bank_id=bank_with_facts, query=QUERY)

    assert result.text == ANSWER


async def test_a_done_that_stays_empty_is_still_an_error(client, llm, bank_with_facts):
    _native_model(llm, lambda _request: ToolCall("done", {"answer_found": True}))

    with pytest.raises(ServiceException) as raised:
        await client.areflect(bank_id=bank_with_facts, query=QUERY)

    assert "returned no answer" in str(raised.value)
