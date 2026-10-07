"""A model that answers in its own schema is corrected, and the retain still lands.

Issue #5280. Behind an OpenAI-compatible relay that accepts ``response_format`` and
silently drops it, the extraction model has nothing left to follow and answers with
fact objects keyed its own way — ``subject``/``predicate``/``object`` where the
extractor needs ``what``. Every fact is discarded, and before this fix the re-ask
could not converge: nothing in a bare retry changed what the model had been told, so
four identical answers became four identical discards and an HTTP 500 that stored
nothing.

This story is the composition the unit suite cannot see: extraction drifting, the
retry carrying a correction, the worker finishing, and the facts being *recallable* at
the end. The drifted reply is returned until the correction actually shows up in the
prompt — so the rule only yields usable facts if the server really sent one, which is
the whole mechanism under test.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from hindsight_system_tests.payloads import consolidation, extracted, fact
from hindsight_system_tests.rulebook import ChatRequest

pytestmark = pytest.mark.asyncio

CONTENT = "Maria is the head of operations. She migrated the billing service to Frankfurt."

# The line the correction puts in the prompt. Matching on it rather than a call counter
# keeps the story honest: a retry that forgot to carry the correction never gets the
# good reply, and the test fails instead of passing on the second attempt regardless.
CORRECTION_MARKER = "Valid keys for each fact"


class DriftedFact(BaseModel):
    """A fact in the shape the relay-served model invented, from the issue's dump."""

    subject: str
    predicate: str
    object: str


class DriftedFacts(BaseModel):
    facts: list[DriftedFact]


DRIFTED = DriftedFacts(
    facts=[
        DriftedFact(subject="Maria", predicate="role", object="head of operations"),
        DriftedFact(subject="Maria", predicate="migrated", object="the billing service to Frankfurt"),
    ]
)

USABLE = extracted(
    fact("Maria is the head of operations", who="Maria", entities=["Maria"]),
    fact(
        "Maria migrated the billing service to Frankfurt",
        who="Maria",
        where="Frankfurt",
        entities=["Maria", "Frankfurt", "billing service"],
    ),
)


async def test_a_drifted_extraction_is_corrected_and_the_facts_survive(
    retrying_client, llm, bank_id, settled
):
    def answer(request: ChatRequest) -> BaseModel:
        # Drift until corrected. The correction arrives as a user turn, which is the
        # half a relay that drops the system message still delivers — so this is also
        # the assertion that it was not put somewhere the model would never see.
        if CORRECTION_MARKER in request.user_text:
            return USABLE
        return DRIFTED

    llm.on_step("extract_facts", contains="Frankfurt").answers_with(answer)
    llm.on_step("consolidate").returns(consolidation())

    # Before the fix this raised: all facts unusable on every attempt, nothing stored.
    await retrying_client.aretain(bank_id=bank_id, content=CONTENT)
    await settled(bank_id)

    response = await retrying_client.arecall(bank_id=bank_id, query="What does Maria do?")

    # Both facts are there, in the corrected shape, rendered from `what` plus the
    # labelled dimensions. Nothing of the drifted reply leaked through: a fact built
    # from `object` would have no `what` to render and never have been stored at all.
    # The order is pinned, not just the membership — with the LLM, embedder and
    # reranker stubbed this ranking is deterministic.
    assert [result.text for result in response.results] == [
        "Maria migrated the billing service to Frankfurt | Involving: Maria",
        "Maria is the head of operations | Involving: Maria",
    ]
    assert all(result.type == "world" for result in response.results)
