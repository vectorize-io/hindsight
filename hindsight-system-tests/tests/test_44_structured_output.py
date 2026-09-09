"""Reflect can return a machine-readable answer alongside the prose one.

An agent that has to regex a paragraph is not integrated, it is guessing. So
reflect takes a JSON Schema and returns `structured_output` shaped to it.

The mechanism is worth knowing because it is not what most people assume: the
schema does *not* constrain the answering turn. Reflect writes prose first, then
makes a **second** extraction call that reshapes that prose into the schema. Two
consequences follow — the prose answer is unchanged by asking for structure, and
structured output costs an extra model call. Both are asserted below.
"""

from __future__ import annotations

import json

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

QUERY = "Where does Alice live?"
ANSWER = "Alice lives in Berlin."

SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "confident": {"type": "boolean"}},
    "required": ["city", "confident"],
    "additionalProperties": False,
}


@pytest.fixture
async def bank_with_facts(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    llm.on_step("reflect").calls_the_offered_tool(query="Alice")
    llm.on_step("reflect_answer").returns_text(ANSWER)
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.")
    await settled(bank_id)
    return bank_id


async def test_the_answer_comes_back_shaped_to_the_schema(client, llm, bank_with_facts):
    llm.on_step("reflect_structured").returns_text(json.dumps({"city": "Berlin", "confident": True}))

    response = await client.areflect(bank_id=bank_with_facts, query=QUERY, response_schema=SCHEMA)

    assert response.structured_output == {"city": "Berlin", "confident": True}


async def test_the_prose_answer_is_unaffected(client, llm, bank_with_facts):
    """The schema shapes an extra field, not the answer. A caller who wants both
    a paragraph for a human and JSON for a machine gets exactly that."""
    llm.on_step("reflect_structured").returns_text(json.dumps({"city": "Berlin", "confident": True}))

    response = await client.areflect(bank_id=bank_with_facts, query=QUERY, response_schema=SCHEMA)

    assert response.text == ANSWER


async def test_the_schema_is_shown_to_the_extraction_call(client, llm, bank_with_facts):
    """It reaches the model that has to satisfy it — the second call, not the
    answering one. A schema that never arrives produces plausible JSON with the
    wrong field names."""
    llm.on_step("reflect_structured").returns_text(json.dumps({"city": "Berlin", "confident": True}))

    await client.areflect(bank_id=bank_with_facts, query=QUERY, response_schema=SCHEMA)

    prompts = llm.prompts_for("reflect_structured")
    assert prompts, "no extraction call was made"
    assert any("confident" in prompt and "city" in prompt for prompt in prompts)


async def test_no_extraction_call_happens_without_a_schema(client, llm, bank_with_facts):
    """Structured output costs a whole extra model call, so nothing pays for it
    by default. Declaring no rule for the extraction step is the assertion: if
    one were made, it would be unscripted and fail the test."""
    response = await client.areflect(bank_id=bank_with_facts, query=QUERY)

    assert response.structured_output is None
    assert llm.prompts_for("reflect_structured") == []


async def test_a_failed_extraction_degrades_but_says_so(client, llm, bank_with_facts):
    """Two halves of one contract.

    Returning the prose rather than failing the whole reflect is right — the
    answer is useful even when the machine-readable half is not. But `null` on
    its own meant three different things (the call errored, the output would not
    parse, or there was genuinely nothing to extract), and only the first two are
    worth retrying or alerting on. #4230 made the failure distinguishable.
    """
    llm.on_step("reflect_structured").returns_text("absolutely not json")

    response = await client.areflect(bank_id=bank_with_facts, query=QUERY, response_schema=SCHEMA)

    assert response.text == ANSWER
    assert response.structured_output is None
    assert response.structured_output_error, "a failed extraction must not look like an empty one"
