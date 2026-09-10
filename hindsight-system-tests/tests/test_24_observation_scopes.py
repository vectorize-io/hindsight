"""One memory, several tags — and control over which observations it feeds.

By default a memory's tags are taken together: one scope, one observation. That
is wrong for the common multi-party case. A lesson transcript tagged
`student:alice` and `teacher:bob` is evidence about *both* of them, and a single
observation tagged with both belongs to neither — recalling Alice's history
would surface an observation that is also about Bob, and vice versa.

`observation_scopes` decides the boundary, and the isolation between scopes is
the load-bearing property: consolidation matches existing observations with
`all_strict`, so a pass scoped to `["student:alice"]` can only ever update an
observation tagged exactly that. Weaken it and one student's observations start
absorbing another's evidence — a privacy failure, not a ranking one.

The last test pins a documented footgun: `[]` means *zero* scopes and silently
falls back to `combined`, while `[[]]` (and `"shared"`) means one global scope.
Two spellings one character apart, opposite meanings, no error either way.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import Consolidation, Observation, extracted, fact, fact_ids_in

pytestmark = pytest.mark.asyncio

STUDENT = "student:alice"
TEACHER = "teacher:bob"
SESSION = "session-id:s1"

CONTENT = "Alice practised scales for an hour."
OBSERVATION = "Alice is practising regularly"


def _observe(text: str):
    """One observation per consolidation pass, over whatever facts it was shown."""

    def build(request) -> Consolidation:
        return Consolidation(
            creates=[Observation(text=text, source_fact_ids=fact_ids_in(request.all_text), reason="system test")]
        )

    return build


@pytest.fixture(autouse=True)
def _extraction(llm):
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice practised scales for an hour", who="Alice", entities=["Alice"]))
    )
    llm.on_step("consolidate").answers_with(_observe(OBSERVATION))


async def _retain(client, bank: str, settled, **item) -> None:
    await client.aretain_batch(bank_id=bank, items=[{"content": CONTENT, **item}])
    await settled(bank)


async def _observation_tags(client, bank: str) -> list[list[str]]:
    memories = await client.memory.list_memories(bank, limit=100)
    return sorted(
        (sorted(m.tags) for m in memories.items if m.fact_type == "observation"),
        key=lambda tags: tags,
    )


async def test_by_default_the_tags_are_one_scope(client, bank_id, settled):
    """`combined`: the memory's tags taken together, one observation carrying
    both — the behaviour the other modes exist to change."""
    await _retain(client, bank_id, settled, tags=[STUDENT, TEACHER])

    assert await _observation_tags(client, bank_id) == [sorted([STUDENT, TEACHER])]


async def test_per_tag_gives_each_tag_its_own_observation(client, bank_id, settled):
    """The multi-party case. Two observations, each tagged with only its own
    party — so Alice's history and Bob's are separately recallable."""
    await _retain(client, bank_id, settled, tags=[STUDENT, TEACHER], observation_scopes="per_tag")

    assert await _observation_tags(client, bank_id) == [[STUDENT], [TEACHER]]


async def test_shared_puts_everything_in_one_untagged_scope(client, bank_id, settled):
    """For tags that are provenance rather than a boundary — a session id you
    want for filtering and debugging, but which must not fragment the
    observations into one per session."""
    await _retain(client, bank_id, settled, tags=[STUDENT, SESSION], observation_scopes="shared")

    assert await _observation_tags(client, bank_id) == [[]]


async def test_an_explicit_scope_list_takes_exactly_those_combinations(client, bank_id, settled):
    """Full control: one pass per inner list, and nothing else."""
    await _retain(
        client, bank_id, settled, tags=[STUDENT, TEACHER, SESSION], observation_scopes=[[STUDENT], [STUDENT, TEACHER]]
    )

    assert await _observation_tags(client, bank_id) == [[STUDENT], sorted([STUDENT, TEACHER])]


async def test_the_bank_lists_the_scopes_it_holds(client, bank_id, settled):
    """So an operator can see how a bank is partitioned without inferring it from
    the observations themselves."""
    await _retain(client, bank_id, settled, tags=[STUDENT, TEACHER], observation_scopes="per_tag")

    scopes = await client.memory.list_observation_scopes(bank_id)

    assert scopes.total == 2
    assert sorted(sorted(s.tags) for s in scopes.scopes) == [[STUDENT], [TEACHER]]
    assert all(s.count == 1 for s in scopes.scopes)


async def test_one_scope_is_recallable_without_the_other(client, bank_id, settled):
    """The isolation property, seen from the read side.

    `exact` matching is what reaches a precise scope: `any_strict` on
    `[STUDENT]` would also return an observation tagged with both, which is the
    cross-party leak the scopes exist to prevent.
    """
    await _retain(client, bank_id, settled, tags=[STUDENT, TEACHER], observation_scopes="per_tag")

    response = await client.arecall(
        bank_id=bank_id, query="How is Alice doing?", types=["observation"], tags=[STUDENT], tags_match="exact"
    )

    assert [sorted(r.tags) for r in response.results] == [[STUDENT]]


async def test_an_empty_scope_list_is_not_the_global_scope(client, bank_id, settled):
    """The documented footgun, pinned because both spellings are silent.

    `[[]]` — a list holding one empty scope — means "one global, untagged scope".
    `[]` means *zero* scopes, which cannot be honoured, so it falls back to
    `combined`. One character apart, opposite results, no error either way: a
    caller who meant "shared" and typed `[]` gets tag-scoped observations and no
    hint that anything was ignored.
    """
    await _retain(client, bank_id, settled, tags=[STUDENT, TEACHER], observation_scopes=[])

    assert await _observation_tags(client, bank_id) == [sorted([STUDENT, TEACHER])], (
        "an empty scope list must fall back to combined, not to the global scope"
    )
