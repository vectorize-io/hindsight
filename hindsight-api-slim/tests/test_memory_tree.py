"""The memory tree groups observations by subject, keeps every one, and is deterministic."""

import random
from datetime import datetime, timezone

from hindsight_api.engine.memory_tree import LEAF_MAX, TreeObservation, build_memory_tree, iter_files


def _bank() -> list[TreeObservation]:
    topics = {
        "redis": ["Redis", "TTL", "cache"],
        "postgres": ["Postgres", "locking", "migration"],
        "gpu": ["GPU", "vLLM", "Gemma"],
    }
    obs = []
    for name, ents in topics.items():
        for i in range(LEAF_MAX):
            # Each observation also names a PR and a commit, which must not drive grouping.
            obs.append(
                TreeObservation(
                    id=f"{name}-{i}",
                    text=f"{name} fact {i}",
                    proof_count=i % 7 + 1,
                    entities=[ents[i % 3], ents[(i + 1) % 3], f"PR #{i}", f"{i:08x}a", "user"],
                    mentioned_at=datetime(2026, 10, 1 + i % 28, tzinfo=timezone.utc),
                )
            )
    obs.append(TreeObservation(id="lonely", text="no entity here", proof_count=1, entities=["#42"]))
    return obs


def test_groups_by_subject_and_keeps_everything():
    obs = _bank()
    tree = build_memory_tree(obs)

    topics = [f for f in iter_files(tree) if f.kind == "topic"]
    placed = [o.id for f in topics for o in f.observations]
    assert sorted(placed) == sorted(o.id for o in obs)  # every observation, exactly once

    for f in topics:
        subjects = {o.id.split("-")[0] for o in f.observations}
        assert len(subjects) == 1, f"{f.path} mixes {subjects}"

    paths = {f.path for f in topics}
    assert "other.md" in paths
    assert not any("pr" in p.split("-") or "user" in p for p in paths)
    assert all(f.path.endswith("README.md") for f in iter_files(tree) if f.kind == "readme")
    # Each line carries its date and source count, so grep sees them.
    assert "(2026-10-01 · 1 sources)" in next(f for f in topics if f.path != "other.md").content


def test_is_deterministic():
    a = build_memory_tree(_bank())
    b = build_memory_tree(_bank())
    assert [(f.path, f.content) for f in iter_files(a)] == [(f.path, f.content) for f in iter_files(b)]


# --- Stability: the tree is rebuilt on every request (no stored tree to explode per scope), so a
# small change to the bank must leave most observations in the file they were in. The floors sit
# just under what the build reaches on this bank and on a real 2.5k-observation bank (1% change:
# 87-97% stay; 5%: 69-93%), so a change that makes the tree less stable fails here.


def _realistic_bank(seed: int = 7, size: int = 2500) -> list[TreeObservation]:
    """A bank shaped like the real ones the tree was tuned on (2.5k-3k observations): subjects of
    very different sizes (a long tail), each with a few core entities used unevenly, tools shared
    across subjects, and about a third of observations carrying only one-off entities (a file,
    a PR title) that nothing else mentions."""
    rng = random.Random(seed)
    subjects = 40
    weights = [1 / (s + 1) for s in range(subjects)]  # Zipf: subject 0 is ~40x subject 39
    tools = [f"tool-{t}" for t in range(15)]
    obs = []
    for i in range(size):
        s = rng.choices(range(subjects), weights)[0]
        ents = [f"oneoff-{i}-{j}" for j in range(rng.randint(1, 3))]
        if rng.random() > 0.33:
            # Core entities used unevenly: the first far more than the last.
            ents += rng.sample([f"subject{s}-{e}" for e in range(4)], rng.choice([1, 1, 2, 3]))
            if rng.random() < 0.3:
                ents.append(rng.choice(tools))
        obs.append(TreeObservation(id=f"o{i:05d}", text=f"subject {s} fact {i}", proof_count=1, entities=ents))
    return obs


def _files_of(obs: list[TreeObservation]) -> dict[str, str]:
    tree = build_memory_tree(obs)
    return {o.id: f.path for f in iter_files(tree) if f.kind == "topic" for o in f.observations}


def _kept(before: dict[str, str], after: dict[str, str]) -> float:
    """Share of the observations in both trees that sit in the same file."""
    common = before.keys() & after.keys()
    return sum(before[i] == after[i] for i in common) / len(common)


def test_input_order_does_not_change_the_tree():
    obs = _realistic_bank()
    shuffled = random.Random(1).sample(obs, len(obs))
    a, b = build_memory_tree(obs), build_memory_tree(shuffled)
    assert [(f.path, f.content) for f in iter_files(a)] == [(f.path, f.content) for f in iter_files(b)]


def test_removing_a_few_observations_keeps_the_rest_in_place():
    obs = _realistic_bank()
    before = _files_of(obs)
    for trial in range(3):
        rng = random.Random(trial)
        assert _kept(before, _files_of(rng.sample(obs, len(obs) - len(obs) // 100))) >= 0.85, f"1%, trial {trial}"
        assert _kept(before, _files_of(rng.sample(obs, len(obs) - len(obs) // 20))) >= 0.65, f"5%, trial {trial}"


def test_adding_a_few_observations_keeps_the_rest_in_place():
    obs = _realistic_bank()
    before = _files_of(obs)
    # 1% new observations from the same kind of bank.
    extra = [
        TreeObservation(id=f"new-{o.id}", text=o.text, proof_count=1, entities=o.entities)
        for o in random.Random(3).sample(_realistic_bank(seed=8), len(obs) // 100)
    ]
    assert _kept(before, _files_of(obs + extra)) >= 0.95


def test_an_observation_that_fits_no_topic_goes_to_other():
    """The real case: a fact about deleting a document's request history, whose only shared
    entity ("document") a few observations in unrelated subjects also carry. It must not be
    forced into a topic."""
    obs = _realistic_bank()
    for o in obs[:: len(obs) // 4][:4]:  # four observations, in different subjects
        o.entities.append("document")
    stray = TreeObservation(
        id="stray", text="delete a document's request history", proof_count=1, entities=["delete_document", "document"]
    )
    where = _files_of(obs + [stray])
    assert where["stray"].endswith("other.md"), where["stray"]
