"""A bank's disposition and mission reach the model that does the reasoning.

Disposition is the soft counterpart to a directive: three traits — skepticism,
literalism, empathy — that shade *how* an agent reasons rather than dictating
what it must do, plus a free-text mission giving it an identity.

Whether a trait produces a measurably different answer is a question about the
model, and belongs with the judge tests. What belongs here is the delivery:
these settings live on the bank, a long way from the reflect call, and a config
value that never reaches the prompt is a knob connected to nothing. The failure
is completely silent — the API accepts the setting, returns it on read, and the
agent behaves exactly as it did before.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests import reflect_loop
from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

QUERY = "Where does Alice live?"
ANSWER = "Alice lives in Berlin."
MISSION = "You are Alice's meticulous relocation assistant."


@pytest.fixture
async def bank_with_facts(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.")
    await settled(bank_id)
    return bank_id


async def test_disposition_traits_round_trip(client, bank_with_facts):
    await client.acreate_bank(
        bank_id=bank_with_facts,
        disposition_skepticism=5,
        disposition_literalism=1,
        disposition_empathy=4,
    )

    listing = await client.banks.list_banks(q=bank_with_facts)
    bank = next(b for b in listing.banks if b.bank_id == bank_with_facts)
    assert bank.disposition.skepticism == 5
    assert bank.disposition.literalism == 1
    assert bank.disposition.empathy == 4


async def test_the_disposition_reaches_the_reasoning_prompt(client, llm, bank_with_facts):
    """The knob has to be connected. Storing it is provable and easy; delivering
    it is the part that can silently stop happening."""
    await client.acreate_bank(bank_id=bank_with_facts, disposition_skepticism=5, disposition_empathy=1)
    reflect_loop(llm, answer=ANSWER)

    await client.areflect(bank_id=bank_with_facts, query=QUERY)

    prompts = llm.prompts_for("reflect")
    assert prompts, "reflect took no search turn"
    assert any("skeptic" in prompt.lower() for prompt in prompts), "disposition never reached the model"


async def test_a_configured_disposition_arrives_as_an_instruction(client, llm, bank_with_facts):
    """Numbers are metadata; the prompt has to say what the number means.

    The test above is satisfied by the bare `Disposition: skepticism=5` line — the
    word "skepticism" contains "skeptic" — and that is exactly the shape a weaker
    model ignores, answering a skepticism=5 bank the way it answers a trusting one.
    """
    await client.acreate_bank(bank_id=bank_with_facts, disposition_skepticism=5)
    reflect_loop(llm, answer=ANSWER)

    await client.areflect(bank_id=bank_with_facts, query=QUERY)

    prompts = llm.prompts_for("reflect")
    assert any("critically examine" in prompt.lower() for prompt in prompts), (
        "the disposition arrived as a number with nothing telling the model what to do with it"
    )


async def test_an_untouched_disposition_adds_no_instructions(client, llm, bank_with_facts):
    """3/3/3 is what a bank that never configured the traits reports, and it must keep
    the prompt it had before instructions existed — nobody pays for a feature they did
    not turn on."""
    await client.acreate_bank(
        bank_id=bank_with_facts,
        disposition_skepticism=3,
        disposition_literalism=3,
        disposition_empathy=3,
    )
    reflect_loop(llm, answer=ANSWER)

    await client.areflect(bank_id=bank_with_facts, query=QUERY)

    prompts = llm.prompts_for("reflect")
    assert any("skepticism=3" in prompt for prompt in prompts), "the traits themselves still belong in the prompt"
    assert not any("your disposition traits" in prompt.lower() for prompt in prompts), (
        "a neutral disposition gained instructions it never asked for"
    )


async def test_opposite_dispositions_produce_different_prompts(client, llm, bank_with_facts):
    """The strongest deterministic statement available without judging output: the
    *instructions* differ. A disposition that reached the prompt as a constant
    string would pass the previous test and fail this one.
    """
    await client.acreate_bank(bank_id=bank_with_facts, disposition_skepticism=5, disposition_literalism=5)
    reflect_loop(llm, answer=ANSWER)
    await client.areflect(bank_id=bank_with_facts, query=QUERY)
    skeptical = llm.prompts_for("reflect")[0]

    llm.reset()
    await client.acreate_bank(bank_id=bank_with_facts, disposition_skepticism=1, disposition_literalism=1)
    reflect_loop(llm, answer=ANSWER)
    await client.areflect(bank_id=bank_with_facts, query=QUERY)
    trusting = llm.prompts_for("reflect")[0]

    assert skeptical != trusting, "changing the disposition changed nothing the model reads"


async def test_the_mission_reaches_the_reasoning_prompt(client, llm, bank_with_facts):
    """A bank's identity is free text and therefore easy to check for verbatim —
    it either arrived or it did not."""
    await client.aset_reflect_mission(bank_id=bank_with_facts, reflect_mission=MISSION)
    reflect_loop(llm, answer=ANSWER)

    await client.areflect(bank_id=bank_with_facts, query=QUERY)

    assert any(MISSION in prompt for prompt in llm.prompts_for("reflect"))
