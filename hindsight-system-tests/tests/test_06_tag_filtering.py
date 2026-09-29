"""Tags scope a recall — and a misspelled one only forgives you if you ask,
and only if it is close enough to earn it.

Three things are worth pinning here.

`tags_match` genuinely changes the result set, including how untagged memories
are treated: `any` admits them, `any_strict` does not. A filter that quietly
matches everything looks identical to a working one until the day it matters.

Tolerance is opt-in. A typo on the plain `tags` parameter is *not* forgiven, no
matter how close — that is deliberate, because a filter that silently widens
itself returns the wrong memories without saying so. Fuzziness lives on a
`tag_groups` leaf with `resolve="fuzzy"` (#4026).

And tolerance has a floor: trigram similarity at 0.45. Short tags are the
surprise — trigrams are scarce in a five-letter word, so a one-letter slip in
`music` scores 0.33 and is *not* forgiven, while the same slip in `typescript`
scores 0.47 and is. Anyone who assumes "one typo is always fine" is wrong in a
way no error message will tell them, so it is asserted.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

QUERY = "What do we know about Alice?"

TYPESCRIPT_FACT = "Alice ships the TypeScript client | Involving: Alice"
MUSIC_FACT = "Alice plays the cello professionally | Involving: Alice"
UNTAGGED_FACT = "Alice speaks Portuguese | Involving: Alice"


@pytest.fixture
async def tagged_bank(client, llm, bank_id, settled) -> str:
    """Three documents: one tagged `typescript`, one `music`, one with no tags."""
    llm.on_step("extract_facts", contains="TypeScript").returns(
        extracted(fact("Alice ships the TypeScript client", who="Alice", entities=["Alice"]))
    )
    llm.on_step("extract_facts", contains="cello").returns(
        extracted(fact("Alice plays the cello professionally", who="Alice", entities=["Alice", "cello"]))
    )
    llm.on_step("extract_facts", contains="Portuguese").returns(
        extracted(fact("Alice speaks Portuguese", who="Alice", entities=["Alice"]))
    )
    llm.on_step("consolidate").returns(consolidation())

    await client.aretain(bank_id=bank_id, content="Alice ships the TypeScript client.", tags=["typescript"])
    await client.aretain(bank_id=bank_id, content="Alice plays the cello.", tags=["music"])
    await client.aretain(bank_id=bank_id, content="Alice speaks Portuguese.")
    await settled(bank_id)
    return bank_id


async def test_a_strict_tag_filter_returns_only_that_tag(client, tagged_bank):
    response = await client.arecall(bank_id=tagged_bank, query=QUERY, tags=["typescript"], tags_match="any_strict")

    assert [r.text for r in response.results] == [TYPESCRIPT_FACT]


async def test_the_non_strict_mode_also_admits_untagged_memories(client, tagged_bank):
    """`any` treats an untagged memory as unclassified rather than excluded — the
    difference between the two modes, and the reason both exist."""
    response = await client.arecall(bank_id=tagged_bank, query=QUERY, tags=["typescript"], tags_match="any")

    assert sorted(r.text for r in response.results) == sorted([TYPESCRIPT_FACT, UNTAGGED_FACT])


async def test_an_unknown_tag_matches_nothing(client, tagged_bank):
    response = await client.arecall(bank_id=tagged_bank, query=QUERY, tags=["accounting"], tags_match="any_strict")

    assert response.results == []


async def test_a_typo_on_the_plain_tags_parameter_is_not_forgiven(client, tagged_bank):
    """Deliberate, not a gap: `tags` is literal. The same typo *is* forgiven on a
    fuzzy tag group below, which is what makes this a choice rather than a bug."""
    response = await client.arecall(bank_id=tagged_bank, query=QUERY, tags=["typsecript"], tags_match="any_strict")

    assert response.results == []


async def test_a_fuzzy_tag_group_reaches_through_the_same_typo(client, tagged_bank):
    """`typsecript` → `typescript`, similarity 0.47 against a 0.45 floor."""
    response = await client.arecall(
        bank_id=tagged_bank,
        query=QUERY,
        tag_groups=[{"tags": ["typsecript"], "match": "any_strict", "resolve": "fuzzy"}],
    )

    assert [r.text for r in response.results] == [TYPESCRIPT_FACT]


async def test_a_short_tag_does_not_survive_the_same_kind_of_typo(client, tagged_bank):
    """The counter-intuitive limit, pinned on purpose.

    `musci` → `music` is one transposition, exactly like `typsecript`, but a
    five-letter word has too few trigrams to score above the floor (0.33). Fuzzy
    matching is not "one typo is fine"; it is a similarity threshold, and short
    tags sit below it.
    """
    response = await client.arecall(
        bank_id=tagged_bank,
        query=QUERY,
        tag_groups=[{"tags": ["musci"], "match": "any_strict", "resolve": "fuzzy"}],
    )

    assert response.results == []


async def test_a_fuzzy_group_still_will_not_invent_a_match(client, tagged_bank):
    """A tag resembling nothing in the bank resolves to nothing, rather than
    drifting onto the nearest survivor."""
    response = await client.arecall(
        bank_id=tagged_bank,
        query=QUERY,
        tag_groups=[{"tags": ["accounting"], "match": "any_strict", "resolve": "fuzzy"}],
    )

    assert response.results == []
