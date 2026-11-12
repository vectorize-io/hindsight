"""The memory tree groups observations by subject, keeps every one, and is deterministic."""

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
