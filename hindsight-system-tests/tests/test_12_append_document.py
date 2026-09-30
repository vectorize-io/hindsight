"""Appending to a document adds to it without re-litigating what is already there.

`update_mode="append"` is the conversation path: each turn extends the same
document rather than replacing it. The distinction from `replace` is the whole
point — an append must *not* run the removal diff, because everything already in
the document is still true, it simply is not in the new fragment.

Getting that backwards is quiet and expensive: appending with replace semantics
deletes the entire history on every turn, and the bank ends up remembering only
the most recent message.
"""

from __future__ import annotations

import json

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

DOCUMENT_ID = "session-alice"

FIRST = "Alice moved to Berlin."
SECOND = "Alice plays cello."

BERLIN = "Alice moved to Berlin | Involving: Alice"
CELLO = "Alice plays cello | Involving: Alice"


async def _fact_texts(client, bank: str) -> list[str]:
    memories = await client.memory.list_memories(bank, limit=100)
    return sorted(item.text for item in memories.items if item.state == "valid")


@pytest.fixture
async def appended(client, llm, bank_id, settled):
    llm.on_step("extract_facts", contains="Berlin").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("extract_facts", contains="cello").returns(
        extracted(fact("Alice plays cello", who="Alice", entities=["Alice", "cello"]))
    )
    llm.on_step("consolidate").returns(consolidation())

    await client.aretain(bank_id=bank_id, content=FIRST, document_id=DOCUMENT_ID)
    await settled(bank_id)
    await client.aretain(bank_id=bank_id, content=SECOND, document_id=DOCUMENT_ID, update_mode="append")
    await settled(bank_id)


async def test_the_earlier_turn_survives_the_later_one(client, bank_id, appended):
    """The failure this guards is total: with replace semantics only the cello
    fact would remain, and every earlier turn of the conversation would be gone."""
    assert await _fact_texts(client, bank_id) == sorted([BERLIN, CELLO])


async def test_the_document_accumulates_the_text(client, bank_id, appended):
    """One document holding both turns, joined — not two documents, and not the
    second turn overwriting the first."""
    document = await client.documents.get_document(bank_id, DOCUMENT_ID)

    assert document.original_text == f"{FIRST}\n{SECOND}"
    assert document.memory_unit_count == 2


async def test_appending_the_same_turn_twice_does_not_duplicate_the_fact(client, bank_id, appended, settled):
    """Re-sending a turn is normal — a retried request, a replayed buffer — and
    must not leave the bank claiming the same thing twice."""
    await client.aretain(bank_id=bank_id, content=SECOND, document_id=DOCUMENT_ID, update_mode="append")
    await settled(bank_id)

    assert await _fact_texts(client, bank_id) == sorted([BERLIN, CELLO])


async def test_oversized_json_conversation_append_preserves_old_and_new_turns(client, llm, bank_id, settled):
    """The splitter and append guard must agree about structural JSON appends.

    A merged array is not a literal byte extension: its closing bracket moves.
    Use a tail larger than the default batch budget to exercise the full-body
    guard through the public client, then append again to check round-tripping.
    """
    berlin = fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"])
    cello = fact("Alice plays cello", who="Alice", entities=["Alice", "cello"])
    llm.on_step("extract_facts", contains="Berlin").returns(extracted(berlin))
    llm.on_step("extract_facts", contains="cello").returns(extracted(cello))
    llm.on_step("extract_facts", contains="Nothing to remember").returns(extracted())
    llm.on_step("consolidate").returns(consolidation())

    first = [{"role": "user", "content": FIRST}]
    # Neutral filler, so the oversized turn states no fact of its own: repeating a fact
    # sentence would extract it once per chunk and blur what the story checks.
    large_tail = [{"role": "user", "content": "Nothing to remember here. " * 4000}]
    final_tail = [{"role": "user", "content": SECOND}]
    await client.aretain(bank_id=bank_id, content=json.dumps(first), document_id=DOCUMENT_ID)
    await settled(bank_id)

    expected = first.copy()
    for tail, facts in ((large_tail, [BERLIN]), (final_tail, [BERLIN, CELLO])):
        await client.aretain(bank_id=bank_id, content=json.dumps(tail), document_id=DOCUMENT_ID, update_mode="append")
        await settled(bank_id)
        expected.extend(tail)
        document = await client.documents.get_document(bank_id, DOCUMENT_ID)
        assert json.loads(document.original_text) == expected
        assert await _fact_texts(client, bank_id) == sorted(facts)
# ---------------------------------------------------------------------------
# An append must also carry the *settings* the turn was retained under, not just
# its text. The document records them as `retain_params`, and a later reprocess
# rebuilds its retain call from exactly that — so a field the append drops is a
# field every reprocess silently substitutes the bank default for.
#
# `strategy` was dropped that way (#4590): the append path rebuilt the first
# content item from a hand-listed subset of the caller's fields, and that item is
# what `retain_params` is read from. Appending onto a NEW document was fine —
# nothing to prepend, so the caller's own item stayed first — which is why a
# single-turn session looked correct and only a real conversation was wrong.
# ---------------------------------------------------------------------------

STRATEGY = "conversation"

# A custom extraction instruction is the crispest signal available here: it is
# substituted into the extraction prompt verbatim, so "did this call run under
# the strategy?" becomes a substring the stub can match. If the strategy is lost,
# the call arrives without it, matches no rule, and `_no_unstubbed_calls` fails
# the test with the prompt that actually arrived.
STRATEGY_MARKER = "Record only what the speaker said about themselves."


@pytest.fixture
async def appended_under_a_strategy(client, llm, bank_id, settled):
    await client.banks.update_bank_config(
        bank_id,
        {
            "updates": {
                "retain_strategies": {
                    STRATEGY: {
                        "retain_extraction_mode": "custom",
                        "retain_custom_instructions": STRATEGY_MARKER,
                    }
                }
            }
        },
    )

    berlin = fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"])
    cello = fact("Alice plays cello", who="Alice", entities=["Alice", "cello"])

    # Declared first because rules match in order, and the two chunkings differ:
    # the append extracts each turn as its own content item, while a reprocess
    # re-reads the stored document as one. Without this rule the joined chunk
    # would match the Berlin rule alone and the cello fact would vanish into what
    # looks like a product bug.
    llm.on_step("extract_facts", contains=[STRATEGY_MARKER, "Berlin", "cello"]).returns(extracted(berlin, cello))
    llm.on_step("extract_facts", contains=[STRATEGY_MARKER, "Berlin"]).returns(extracted(berlin))
    llm.on_step("extract_facts", contains=[STRATEGY_MARKER, "cello"]).returns(extracted(cello))
    llm.on_step("consolidate").returns(consolidation())

    for turn, mode in ((FIRST, None), (SECOND, "append")):
        item = {"content": turn, "document_id": DOCUMENT_ID, "strategy": STRATEGY}
        if mode:
            item["update_mode"] = mode
        await client.aretain_batch(bank_id=bank_id, items=[item])
        await settled(bank_id)


async def test_an_append_records_the_strategy_it_was_retained_under(client, bank_id, appended_under_a_strategy):
    """`retain_params` is the contract a reprocess replays from. Losing the
    strategy here is invisible until someone reprocesses and quietly gets the
    bank default."""
    document = await client.documents.get_document(bank_id, DOCUMENT_ID)

    assert (document.retain_params or {}).get("strategy") == STRATEGY


async def test_reprocessing_an_appended_document_re_extracts_under_that_strategy(
    client, bank_id, appended_under_a_strategy, settled
):
    """The symptom the dropped field actually produces. Every extraction rule
    requires the strategy's marker, so a reprocess that fell back to the bank
    default cannot match either — it fails as an unstubbed call rather than
    quietly re-extracting the document under settings nobody asked for."""
    await client.documents.reprocess_document(bank_id, DOCUMENT_ID)
    await settled(bank_id)

    assert await _fact_texts(client, bank_id) == sorted([BERLIN, CELLO])
