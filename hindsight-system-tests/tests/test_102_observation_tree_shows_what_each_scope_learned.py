"""The observation tree shows what consolidation learned, scope by scope, as it is now.

The tree is built on request from the bank's observations, so it composes two things
nothing else tests together: consolidation's tag scopes (an observation carries exactly
its scope's tags) and the tree's own tag filter (the same `tags` / `tags_match` recall
takes). Two teams share a bank, each learning about four subjects; each team's tree must
group its own observations and hold none of the other's, and the whole-bank tree must
file every observation exactly once.

How a bank too big for one file is split into topics is the unit tests' job
(hindsight-api-slim/tests/test_observation_tree.py); a story needs a bank past 120 observations
for that, so this one stays small and covers the hand-off between the steps instead.

Then consolidation rewrites an observation. The tree is not stored, so the next request
must show the new text, in the same file, rather than a copy that went stale when the
observation changed underneath it.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import (
    Consolidation,
    Observation,
    ObservationUpdate,
    batch_fact_id,
    extracted,
    fact,
    fact_ids_in,
)

pytestmark = pytest.mark.asyncio

TEAM_A = "team:a"
TEAM_B = "team:b"
# Four subjects of two facts each: an entity on two observations is a topic (one on a
# single observation names nothing), and on a quarter of a scope it is not "everywhere"
# (the tree ignores entities on over a third; three equal subjects would sit exactly there).
LEARNED = {
    TEAM_A: {
        "Redis keys expire after an hour": "Redis",
        "Redis runs in cluster mode": "Redis",
        "Postgres holds the bank rows": "Postgres",
        "Postgres replicates to a standby": "Postgres",
        "Kafka carries the retain events": "Kafka",
        "Kafka keeps events for a week": "Kafka",
        "Nginx terminates TLS for the API": "Nginx",
        "Nginx rate-limits the retain route": "Nginx",
    },
    TEAM_B: {
        "The GPU nodes run vLLM": "GPU",
        "The GPU nodes are H100s": "GPU",
        "Helm deploys the inference stack": "Helm",
        "Helm charts live in the deploy repo": "Helm",
        "Grafana shows the token throughput": "Grafana",
        "Grafana alerts on queue depth": "Grafana",
        "Triton serves the reranker": "Triton",
        "Triton batches requests by length": "Triton",
    },
}
REWRITTEN = "Redis keys expire after a day"


def _files(folder) -> list:
    return list(folder.files) + [f for sub in folder.folders for f in _files(sub)]


def _placed(tree) -> dict[str, str]:
    """Observation text -> the topic file listing it."""
    return {o.text: f.path for f in _files(tree) if f.kind == "topic" for o in f.observations}


def _one_observation_per_fact(request) -> Consolidation:
    return Consolidation(
        creates=[
            Observation(text=text, source_fact_ids=[fid], reason="system test")
            for facts in LEARNED.values()
            for text in facts
            if (fid := batch_fact_id(request, text))
        ]
    )


async def _two_teams(client, llm, bank_id, settled) -> None:
    for team, facts in LEARNED.items():
        llm.on_step("extract_facts", contains=team).returns(
            extracted(*(fact(text, entities=[entity]) for text, entity in facts.items()))
        )
    llm.on_step("consolidate").answers_with(_one_observation_per_fact)
    for team in LEARNED:
        await client.aretain(bank_id=bank_id, content=f"Notes from {team}", tags=[team])
        await settled(bank_id)


async def test_each_team_sees_only_its_own_observations(client, llm, bank_id, settled):
    await _two_teams(client, llm, bank_id, settled)

    team_a = await client.memory.get_observation_tree(bank_id, tags=[TEAM_A], tags_match="any_strict")
    team_b = await client.memory.get_observation_tree(bank_id, tags=[TEAM_B], tags_match="any_strict")
    everything = await client.memory.get_observation_tree(bank_id)

    assert sorted(_placed(team_a)) == sorted(LEARNED[TEAM_A])
    assert sorted(_placed(team_b)) == sorted(LEARNED[TEAM_B])
    assert sorted(_placed(everything)) == sorted([*LEARNED[TEAM_A], *LEARNED[TEAM_B]])
    assert everything.observation_count == 16
    # Grouped, not dumped in misc: a scope this small is its index plus its one topic file.
    for tree in (team_a, team_b, everything):
        assert sorted(f.path for f in _files(tree)) == ["all.md", "index.md"]
    # Every topic file is an Open Knowledge Format concept: frontmatter with a type.
    for f in _files(everything):
        if f.kind == "topic":
            assert f.content.startswith("---\ntype: Memory Topic\n"), f.content[:80]


async def test_a_rewritten_observation_shows_its_new_text_in_the_same_file(client, llm, bank_id, settled):
    await _two_teams(client, llm, bank_id, settled)
    before = _placed(await client.memory.get_observation_tree(bank_id))
    memories = await client.memory.list_memories(bank_id, limit=100)
    redis = next(
        m.id for m in memories.items if m.fact_type == "observation" and m.text == "Redis keys expire after an hour"
    )

    llm.reset()
    llm.on_step("extract_facts").returns(extracted(fact("Redis keys now expire after a day", entities=["Redis"])))

    def rewrite(request) -> Consolidation:
        return Consolidation(
            updates=[
                ObservationUpdate(text=REWRITTEN, observation_id=redis, source_fact_ids=fact_ids_in(request.all_text))
            ]
        )

    llm.on_step("consolidate").answers_with(rewrite)
    await client.aretain(bank_id=bank_id, content="Redis cache update", tags=[TEAM_A])
    await settled(bank_id)

    after = _placed(await client.memory.get_observation_tree(bank_id))
    assert REWRITTEN in after and "Redis keys expire after an hour" not in after
    # Same file as before the rewrite, and every other observation stayed put.
    assert after[REWRITTEN] == before["Redis keys expire after an hour"] == "all.md"
    assert {t: p for t, p in after.items() if t != REWRITTEN} == {
        t: p for t, p in before.items() if t != "Redis keys expire after an hour"
    }
