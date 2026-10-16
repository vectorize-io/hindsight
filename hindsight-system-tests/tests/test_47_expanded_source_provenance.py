"""A memory reflect read through `expand` is evidence like one it recalled (#5474).

Reflect's recall is scoped to world facts, so it returns only Alice's move. The
agent then expands an experience fact it already knew the id of — the flight it
booked — answers from it, and cites it. That citation has to survive: the answer
rests on the flight, so `based_on` must name the flight and not the move.

The seam is between two tools. Citations were checked against what recall
returned, so an id first seen through `expand` was dropped; with nothing valid
left, `based_on` fell back to every recalled fact and credited the move, which
the answer never used.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests import reflect_loop
from hindsight_system_tests.payloads import consolidation, extracted, fact
from hindsight_system_tests.rulebook import ChatRequest, LLMStub, ToolCall

pytestmark = pytest.mark.asyncio

MOVE = "Alice moved to Berlin in 2021"
FLIGHT = "The agent booked Alice a flight to Munich"
ANSWER = "The agent booked Alice a flight to Munich."


@pytest.fixture
async def bank(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts", contains="Berlin").returns(
        extracted(fact(MOVE, who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("extract_facts", contains="Munich").returns(
        extracted(fact(FLIGHT, fact_type="assistant", who="Alice", entities=["Alice", "Munich"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin in 2021.", document_id="move")
    await client.aretain(bank_id=bank_id, content="The agent booked Alice a flight to Munich.", document_id="flight")
    await settled(bank_id)
    return bank_id


def _expand_then_cite(llm: LLMStub, memory_id: str) -> None:
    def build(request: ChatRequest) -> ToolCall:
        expanded = any(
            call.get("function", {}).get("name") == "expand"
            for m in request.messages
            for call in (m.get("tool_calls") or [])
        )
        if expanded:
            return ToolCall("done", {"answer": ANSWER, "memory_ids": [memory_id]})
        return ToolCall("expand", {"memory_ids": [memory_id], "depth": "document"})

    llm.on_step("reflect", tool="expand").answers_with_tool_call(build)
    # The answer is written by the closing step, which offers only `done`: it cites the flight too.
    llm.on_step("reflect", tool="done").returns_tool_call("done", answer=ANSWER, memory_ids=[memory_id])
    reflect_loop(llm, answer=ANSWER, query="Alice")


async def test_an_expanded_memory_is_cited_in_based_on(client, llm, bank):
    experiences = await client.arecall(bank_id=bank, query="Alice", types=["experience"])
    flight_id = next(r.id for r in experiences.results if FLIGHT in r.text)
    _expand_then_cite(llm, flight_id)

    response = await client.areflect(
        bank_id=bank, query="Where is Alice flying?", fact_types=["world"], include_facts=True
    )

    assert response.text == ANSWER
    cited = [(m.id, m.type) for m in response.based_on.memories]
    assert cited == [(flight_id, "experience")], f"expected only the flight, got {response.based_on.memories}"
