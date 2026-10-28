"""A bank set to the fact-by-fact consolidation format keeps its observations true the same way.

``consolidation_format = "facts"`` changes what the model is asked for: one entry per new fact,
judging every observation that mentions it (repeats / changes / ends / unrelated), instead of
lists of creates and updates. It is written for small models, which lose track of the lists.
The bank must not notice the difference: a job change still ends the old observation in place
and records the new one, and chatter still records nothing — the answer is read back into the
same writes, through the same API a client sees.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import (
    FactConsolidation,
    FactEntry,
    ObservationJudgement,
    batch_fact_id,
    batch_fact_ids,
    extracted,
    fact,
    shown_observations,
)

pytestmark = pytest.mark.asyncio

AT_GOOGLE = "Alice works at Google."
LEFT_GOOGLE = "Alice worked at Google until she left in April 2026."
AT_MICROSOFT = "Alice works at Microsoft."


async def _observations(client, bank: str) -> list[str]:
    memories = await client.memory.list_memories(bank, limit=100)
    return sorted(m.text for m in memories.items if m.fact_type == "observation")


@pytest.fixture
async def fact_format_bank(client, llm, bank_id, settled) -> str:
    await client.banks.update_bank_config(bank_id, {"updates": {"consolidation_format": "facts"}})

    llm.on_step("extract_facts", contains="Google").returns(
        extracted(fact("Alice works at Google", who="Alice", entities=["Alice", "Google"]))
    )
    llm.on_step("consolidate").answers_with(
        lambda request: FactConsolidation(
            facts=[FactEntry(fact_id=fid, about="Alice", adds="yes", new=[AT_GOOGLE]) for fid in batch_fact_ids(request)]
        )
    )
    await client.aretain(bank_id=bank_id, content="Alice works at Google.")
    await settled(bank_id)

    llm.reset()
    llm.on_step("extract_facts", contains="Microsoft").returns(
        extracted(
            fact("Alice left Google in April 2026 and joined Microsoft", who="Alice", entities=["Alice", "Microsoft"]),
            fact("The user said thanks for the help", who="the user"),
        )
    )

    def judge(request) -> FactConsolidation:
        google = shown_observations(request)[AT_GOOGLE]
        return FactConsolidation(
            facts=[
                FactEntry(
                    fact_id=batch_fact_id(request, "Microsoft") or "",
                    about="Alice",
                    existing=[
                        ObservationJudgement(
                            observation_id=google, observation_text=AT_GOOGLE, effect="ends", text=LEFT_GOOGLE
                        )
                    ],
                    adds="yes",
                    new=[AT_MICROSOFT],
                ),
                FactEntry(fact_id=batch_fact_id(request, "thanks") or "", about="the user", kind="momentary"),
            ]
        )

    llm.on_step("consolidate").answers_with(judge)
    await client.aretain(bank_id=bank_id, content="Alice left Google and joined Microsoft. Thanks for the help!")
    await settled(bank_id)
    return bank_id


async def test_an_ended_observation_is_rewritten_and_the_new_one_recorded(client, fact_format_bank):
    assert await _observations(client, fact_format_bank) == sorted([LEFT_GOOGLE, AT_MICROSOFT])


async def test_chatter_records_nothing_and_is_not_asked_again(client, llm, fact_format_bank):
    """The momentary fact is the model's answer, not an oversight: no observation cites it,
    and no further consolidation call asks about it."""
    assert len(llm.prompts_for("consolidate")) == 1
