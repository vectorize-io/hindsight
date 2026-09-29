"""A bank holding a handbook AND the conversations about it can rank the two.

This is the composition nothing else covers. Tag filtering (test_06) proves a
recall can be scoped. Reflect (test_40) proves the loop searches before it
speaks. Neither says what happens when one bank holds both kinds of memory and
they disagree — which is the ordinary state of a bank someone has been using:
the guide says two approvals, a thread says one is enough for small ones, and
after extraction both are flat assertions that retrieve equally well. The chatter
is usually the NEWER of the two, so reflect's recency rule actively prefers it.

What separates them is where each document came from, which a bank records in the
document's metadata. So that metadata has to reach the model — until it did not,
and "prefer the handbook over the chatter" was unexpressible. With it there, an
operator states the precedence in the bank's reflect mission and needs no new
configuration.

Asserted here: the stamp survives into the tool result the model reads, on the
raw-fact path and the observation path both. Whether the model then FOLLOWS a
precedence rule is a question for a real model, and is measured in
`hindsight-system-evals/evals/test_07_source_priority.py`.
"""

from __future__ import annotations

import json

import pytest

from hindsight_system_tests import reflect_loop
from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

GUIDE = {"source": "guide"}
TALK = {"source": "conversation"}

QUERY = "How many approvals does a pull request need?"
#: What the stubbed loop searches for. The stub's embeddings are lexical, so a
#: search phrased in words the facts do not contain returns nothing to rank.
SEARCH = "pull request approvals"
ANSWER = "Two, per the handbook."

POLICY = "A pull request needs two approvals"
HEARSAY = "One approval is enough for a small pull request"


@pytest.fixture
async def mixed_bank(client, llm, bank_id, settled) -> str:
    """One policy sentence from the handbook, one claim from a thread."""
    llm.on_step("extract_facts", contains="two approvals").returns(extracted(fact(POLICY)))
    llm.on_step("extract_facts", contains="One approval").returns(extracted(fact(HEARSAY)))
    llm.on_step("consolidate").returns(consolidation())

    await client.aretain(bank_id=bank_id, content=f"{POLICY}.", metadata=GUIDE)
    await client.aretain(bank_id=bank_id, content=f"{HEARSAY}.", metadata=TALK)
    await settled(bank_id)
    return bank_id


def _recalled(response) -> list[dict]:
    """The memories reflect's recall tool handed the model."""
    for call in (response.trace.tool_calls if response.trace else None) or []:
        if call.tool == "recall":
            return json.loads(json.dumps(call.output or {}, default=str)).get("memories", [])
    raise AssertionError("reflect never called recall")


async def test_the_provenance_stamp_reaches_the_model(client, llm, mixed_bank):
    """Without this, no amount of prompting can rank the two sources."""
    reflect_loop(llm, answer=ANSWER, query=SEARCH)

    response = await client.areflect(bank_id=mixed_bank, query=QUERY, include_tool_calls=True)

    memories = _recalled(response)
    assert memories, "recall returned nothing"

    # Every fact, not just the handbook's: a stamp that survives on one document
    # and is lost on another ranks the sources wrongly rather than not at all.
    stamped = {memory["text"]: memory.get("metadata") for memory in memories}
    expected = {POLICY: GUIDE, HEARSAY: TALK}
    for text, metadata in stamped.items():
        want = next((m for fragment, m in expected.items() if fragment in text), None)
        assert want is not None, f"recall returned a fact this test did not seed: {text!r}"
        assert metadata == want, f"{text!r} lost its provenance: got {metadata!r}, want {want!r}"
    assert any(POLICY in text for text in stamped), f"the handbook fact never came back: {stamped}"


async def test_a_document_with_no_metadata_sends_none(client, llm, bank_id, settled):
    """The cost is paid only by banks that stamp something.

    ``_prune_nulls`` drops an empty bag, so surfacing metadata charges nothing to
    a bank that records none — which is what makes sending it at all defensible.
    """
    llm.on_step("extract_facts").returns(extracted(fact(POLICY)))
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content=f"{POLICY}.")
    await settled(bank_id)
    reflect_loop(llm, answer=ANSWER, query=SEARCH)

    response = await client.areflect(bank_id=bank_id, query=QUERY, include_tool_calls=True)

    assert all("metadata" not in memory for memory in _recalled(response))
