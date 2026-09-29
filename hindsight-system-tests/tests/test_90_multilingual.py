"""Non-Latin content survives the whole round trip.

The model handles the language; the pipeline around it handles the bytes, and
that is where non-English content actually breaks. Text is chunked, hashed,
embedded, tokenised for BM25, stored, and rendered back — and several of those
steps have historically had opinions about what a character is. A tokeniser that
splits CJK per byte, a chunker that cuts mid-codepoint, a full-text index that
strips anything non-ASCII: each produces mangled text or an empty recall, not an
error.

Mixed-script content is the sharper case, because a Chinese sentence containing
an English product name has boundaries in the middle of it — the exact place a
naive ASCII-boundary rule goes wrong.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

CHINESE_CONTENT = "爱丽丝二零二一年搬到柏林。"
CHINESE_FACT = "爱丽丝搬到了柏林"

MIXED_CONTENT = "爱丽丝在 Vectorize 公司使用 TypeScript 开发。"
MIXED_FACT = "爱丽丝在 Vectorize 使用 TypeScript"

EMOJI_CONTENT = "Alice moved to Berlin 🎉 and plays the cello 🎻"
EMOJI_FACT = "Alice celebrated moving to Berlin 🎉🎻"


async def test_chinese_content_is_stored_and_recalled_unmangled(client, llm, bank_id, settled):
    llm.on_step("extract_facts").returns(extracted(fact(CHINESE_FACT, who="爱丽丝", entities=["爱丽丝", "柏林"])))
    llm.on_step("consolidate").returns(consolidation())

    await client.aretain(bank_id=bank_id, content=CHINESE_CONTENT, document_id="d1")
    await settled(bank_id)

    # Byte-for-byte, not merely "contains something Chinese": a chunker that cut
    # mid-codepoint or a store that round-tripped through the wrong encoding would
    # produce text that still looks plausible at a glance.
    document = await client.documents.get_document(bank_id, "d1")
    assert document.original_text == CHINESE_CONTENT

    response = await client.arecall(bank_id=bank_id, query="爱丽丝住在哪里？")
    assert [r.text for r in response.results] == [f"{CHINESE_FACT} | Involving: 爱丽丝"]


async def test_chinese_entities_are_recorded_under_their_own_names(client, llm, bank_id, settled):
    """Entity names are keys — for resolution, for the graph, for co-occurrence.
    A name that loses characters loses the link, silently."""
    llm.on_step("extract_facts").returns(extracted(fact(CHINESE_FACT, who="爱丽丝", entities=["爱丽丝", "柏林"])))
    llm.on_step("consolidate").returns(consolidation())

    await client.aretain(bank_id=bank_id, content=CHINESE_CONTENT)
    await settled(bank_id)

    listing = await client.entities.list_entities(bank_id)
    assert sorted(e.canonical_name for e in listing.items) == sorted(["爱丽丝", "柏林"])


async def test_mixed_script_content_keeps_its_latin_names_intact(client, llm, bank_id, settled):
    """The ASCII-boundary case. English proper nouns embedded in Chinese text sit
    exactly where a naive word-boundary rule mis-splits, and the failure looks
    like a subtly wrong product name rather than an error."""
    llm.on_step("extract_facts").returns(
        extracted(fact(MIXED_FACT, who="爱丽丝", entities=["爱丽丝", "Vectorize", "TypeScript"]))
    )
    llm.on_step("consolidate").returns(consolidation())

    await client.aretain(bank_id=bank_id, content=MIXED_CONTENT, document_id="d1")
    await settled(bank_id)

    assert (await client.documents.get_document(bank_id, "d1")).original_text == MIXED_CONTENT

    listing = await client.entities.list_entities(bank_id)
    names = {e.canonical_name for e in listing.items}
    assert "Vectorize" in names
    assert "TypeScript" in names


async def test_a_latin_query_finds_the_name_inside_non_latin_text(client, llm, bank_id, settled):
    """Searching for the English name has to reach the Chinese sentence holding
    it — otherwise the two scripts are effectively separate banks."""
    llm.on_step("extract_facts").returns(
        extracted(fact(MIXED_FACT, who="爱丽丝", entities=["爱丽丝", "Vectorize", "TypeScript"]))
    )
    llm.on_step("consolidate").returns(consolidation())

    await client.aretain(bank_id=bank_id, content=MIXED_CONTENT)
    await settled(bank_id)

    response = await client.arecall(bank_id=bank_id, query="TypeScript")
    assert [r.text for r in response.results] == [f"{MIXED_FACT} | Involving: 爱丽丝"]


async def test_astral_plane_characters_survive(client, llm, bank_id, settled):
    """Emoji are four-byte codepoints and surrogate pairs in some encodings — the
    other place a length calculation or a truncation quietly corrupts text."""
    llm.on_step("extract_facts").returns(extracted(fact(EMOJI_FACT, who="Alice", entities=["Alice", "Berlin"])))
    llm.on_step("consolidate").returns(consolidation())

    await client.aretain(bank_id=bank_id, content=EMOJI_CONTENT, document_id="d1")
    await settled(bank_id)

    assert (await client.documents.get_document(bank_id, "d1")).original_text == EMOJI_CONTENT

    response = await client.arecall(bank_id=bank_id, query="Where did Alice move?")
    assert [r.text for r in response.results] == [f"{EMOJI_FACT} | Involving: Alice"]
