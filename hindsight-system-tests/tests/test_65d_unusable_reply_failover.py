"""A chain's primary answering retain with the JSON schema still gets the document saved.

Gemma 4 on vLLM, writing retain without a decoding constraint and shown the response
schema, sometimes answers with that schema (``{"$defs": ...}``) instead of the facts. To
the transport it is a normal reply, so the failover chain never reached its next member,
and with no retry budget there was no re-ask: the whole document failed to retain.

The server here chains two stub models with retries at 0. Member 0 always echoes the
schema; member 1 returns the facts. The retain must succeed on member 1's facts, and
member 0 must not be cooled down for it: the next document starts on it again.
"""

from __future__ import annotations

import pytest
from pydantic import RootModel

from hindsight_system_tests import wait_until_settled
from hindsight_system_tests.payloads import consolidation, extracted, fact
from hindsight_system_tests.rulebook import ChatRequest
from hindsight_system_tests.steps import anchor_for

pytestmark = pytest.mark.asyncio

FIRST = "Odalys Brenwick repairs harpsichords in a workshop in Trieste."
SECOND = "Teodor Vasquez tunes church organs across Ljubljana."

SCHEMA_ECHO = RootModel[dict]({"$defs": {"ExtractedFact": {"properties": {"what": {"type": "string"}}}}})


def _by_member(facts_reply):
    """Member 0 (``stub-echo``) echoes the schema; member 1 answers with the facts."""

    def build(request: ChatRequest):
        return SCHEMA_ECHO if request.model == "stub-echo" else facts_reply

    return build


async def test_a_schema_echo_is_answered_by_the_next_member(echo_primary_client, llm, bank_id):
    llm.on_step("extract_facts", contains="Odalys").answers_with(
        _by_member(
            extracted(fact("Odalys Brenwick repairs harpsichords in Trieste", where="Trieste", who="Odalys Brenwick"))
        )
    )
    llm.on_step("extract_facts", contains="Teodor").answers_with(
        _by_member(
            extracted(fact("Teodor Vasquez tunes church organs in Ljubljana", where="Ljubljana", who="Teodor Vasquez"))
        )
    )
    # The retain enqueues consolidation; another server's worker may claim it.
    llm.on_step("consolidate").returns(consolidation())

    await echo_primary_client.aretain(bank_id=bank_id, content=FIRST)
    await echo_primary_client.aretain(bank_id=bank_id, content=SECOND)
    await wait_until_settled(echo_primary_client, bank_id)

    response = await echo_primary_client.arecall(bank_id=bank_id, query="Who repairs or tunes instruments?")
    assert sorted(result.text for result in response.results) == [
        "Odalys Brenwick repairs harpsichords in Trieste | Where: Trieste | Involving: Odalys Brenwick",
        "Teodor Vasquez tunes church organs in Ljubljana | Where: Ljubljana | Involving: Teodor Vasquez",
    ]
    # Each document went to member 0 first, then member 1: the echo did not cool member 0
    # down, so the second document still started on it.
    anchor = anchor_for("extract_facts")
    extraction_calls = [(call.model, "Odalys" in call.all_text) for call in llm.calls if anchor in call.all_text]
    assert extraction_calls == [("stub-echo", True), ("stub-model", True), ("stub-echo", False), ("stub-model", False)]
