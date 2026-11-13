"""The memory tree shows what consolidation learned, scope by scope, as it is now.

The tree is built on request from the bank's observations, so it composes two things
nothing else tests together: consolidation's tag scopes (an observation carries exactly
its scope's tags) and the tree's own tag filter (the same `tags` / `tags_match` recall
takes). Two teams share a bank; each team's tree must hold its own observations and none
of the other's, and the whole-bank tree must file every observation exactly once.

Then consolidation rewrites an observation. The tree is not stored, so the next request
must show the new text, in the same file, rather than a copy that went stale when the
observation changed underneath it.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import Consolidation, ObservationUpdate, extracted, fact, fact_ids_in, observes

pytestmark = pytest.mark.asyncio

TEAM_A = "team:a"
TEAM_B = "team:b"
REDIS = "Redis keys expire after an hour"
GPU = "The GPU nodes run vLLM"


def _files(folder) -> list:
    return list(folder.files) + [f for sub in folder.folders for f in _files(sub)]


def _placed(tree) -> dict[str, str]:
    """Observation text -> the topic file listing it."""
    return {o.text: f.path for f in _files(tree) if f.kind == "topic" for o in f.observations}


async def _two_teams(client, llm, bank_id, settled) -> None:
    llm.on_step("extract_facts", contains="Redis").returns(
        extracted(fact("Redis keys expire after an hour", entities=["Redis"]))
    )
    llm.on_step("extract_facts", contains="GPU").returns(extracted(fact("The GPU nodes run vLLM", entities=["GPU"])))
    llm.on_step("consolidate", contains="Redis").answers_with(observes(REDIS))
    llm.on_step("consolidate", contains="GPU").answers_with(observes(GPU))
    await client.aretain(bank_id=bank_id, content="Redis cache notes", tags=[TEAM_A])
    await client.aretain(bank_id=bank_id, content="GPU cluster notes", tags=[TEAM_B])
    await settled(bank_id)


async def test_each_team_sees_only_its_own_observations(client, llm, bank_id, settled):
    await _two_teams(client, llm, bank_id, settled)

    team_a = await client.memory.get_memory_tree(bank_id, tags=[TEAM_A], tags_match="any_strict")
    team_b = await client.memory.get_memory_tree(bank_id, tags=[TEAM_B], tags_match="any_strict")
    everything = await client.memory.get_memory_tree(bank_id)

    assert list(_placed(team_a)) == [REDIS]
    assert list(_placed(team_b)) == [GPU]
    assert sorted(_placed(everything)) == sorted([REDIS, GPU])
    assert everything.observation_count == 2
    # Every topic file is an Open Knowledge Format concept: frontmatter with a type.
    for f in _files(everything):
        if f.kind == "topic":
            assert f.content.startswith("---\ntype: Memory Topic\n"), f.content[:80]


async def test_a_rewritten_observation_shows_its_new_text_in_the_same_file(client, llm, bank_id, settled):
    await _two_teams(client, llm, bank_id, settled)
    before = _placed(await client.memory.get_memory_tree(bank_id))
    memories = await client.memory.list_memories(bank_id, limit=100)
    redis = next(m.id for m in memories.items if m.fact_type == "observation" and m.text == REDIS)

    llm.reset()
    llm.on_step("extract_facts").returns(extracted(fact("Redis keys now expire after a day", entities=["Redis"])))

    def rewrite(request) -> Consolidation:
        return Consolidation(
            updates=[
                ObservationUpdate(
                    text="Redis keys expire after a day",
                    observation_id=redis,
                    source_fact_ids=fact_ids_in(request.all_text),
                )
            ]
        )

    llm.on_step("consolidate").answers_with(rewrite)
    await client.aretain(bank_id=bank_id, content="Redis cache update", tags=[TEAM_A])
    await settled(bank_id)

    after = _placed(await client.memory.get_memory_tree(bank_id))
    rewritten = [text for text in after if text.startswith("Redis keys expire after a day")]
    assert len(rewritten) == 1, after
    assert REDIS not in after
    assert after[rewritten[0]] == before[REDIS]
    assert after[GPU] == before[GPU]
