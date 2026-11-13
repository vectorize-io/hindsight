"""The observation tree groups observations by subject, keeps every one, and is deterministic."""

import json
import random
import re
from datetime import datetime, timezone

from hindsight_api.engine.observation_tree import (
    LEAF_MAX,
    OKF_VERSION,
    TreeObservation,
    build_observation_tree,
    iter_files,
)


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
    tree = build_observation_tree(obs)

    topics = [f for f in iter_files(tree) if f.kind == "topic"]
    placed = [o.id for f in topics for o in f.observations]
    assert sorted(placed) == sorted(o.id for o in obs)  # every observation, exactly once

    for f in topics:
        subjects = {o.id.split("-")[0] for o in f.observations}
        assert len(subjects) == 1, f"{f.path} mixes {subjects}"

    paths = {f.path for f in topics}
    assert "misc.md" in paths
    assert not any("pr" in p.split("-") or "user" in p for p in paths)
    assert all(f.path.endswith("index.md") for f in iter_files(tree) if f.kind == "index")
    # Each line carries its date and source count, so grep sees them.
    assert "(2026-10-01 00:00 · 1 source)" in next(f for f in topics if f.path != "misc.md").content


def test_is_deterministic():
    a = build_observation_tree(_bank())
    b = build_observation_tree(_bank())
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
    tree = build_observation_tree(obs)
    return {o.id: f.path for f in iter_files(tree) if f.kind == "topic" for o in f.observations}


def _kept(before: dict[str, str], after: dict[str, str]) -> float:
    """Share of the observations in both trees that sit in the same file."""
    common = before.keys() & after.keys()
    return sum(before[i] == after[i] for i in common) / len(common)


def test_input_order_does_not_change_the_tree():
    obs = _realistic_bank()
    shuffled = random.Random(1).sample(obs, len(obs))
    a, b = build_observation_tree(obs), build_observation_tree(shuffled)
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


def test_an_observation_that_fits_no_topic_goes_to_misc():
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
    assert where["stray"].rsplit("/", 1)[-1].startswith("misc"), where["stray"]


def test_misc_is_a_file_never_a_folder():
    """No folder called misc (no misc/misc/misc.md): a folder's leftovers are misc.md, or one
    misc-YYYY-MM.md per month once there are more than fit in one file."""
    tree = build_observation_tree(_realistic_bank())
    paths = [f.path for f in iter_files(tree)]
    assert not any("misc/" in p for p in paths), [p for p in paths if "misc/" in p]
    misc = [p.rsplit("/", 1)[-1] for p in paths if p.rsplit("/", 1)[-1].startswith("misc")]
    assert misc
    assert all(name == "misc.md" or re.fullmatch(r"misc-(\d{4}-(\d{2}|w\d{2})|undated)\.md", name) for name in misc), (
        misc
    )


def test_a_big_misc_is_split_by_month():
    """Months, not misc-1/misc-2: an observation's month never changes, so it never moves file."""
    obs = [
        TreeObservation(
            id=f"m{i}",
            text=f"stray {i}",
            proof_count=1,
            entities=[f"oneoff-{i}"],
            mentioned_at=datetime(2026, 9 + i % 3, 1 + i % 28, tzinfo=timezone.utc),
        )
        for i in range(LEAF_MAX * 2)
    ]
    names = sorted(f.path for f in iter_files(build_observation_tree(obs)) if f.kind == "topic")
    assert names == ["misc-2026-09.md", "misc-2026-10.md", "misc-2026-11.md"]


def test_a_month_too_big_for_one_file_is_split_by_week():
    obs = [
        TreeObservation(
            id=f"m{i}",
            text=f"stray {i}",
            proof_count=1,
            entities=[f"oneoff-{i}"],
            mentioned_at=datetime(2026, 11, 2 + i % 14, tzinfo=timezone.utc),  # Mon 2 .. Sun 15 Nov
        )
        for i in range(LEAF_MAX * 2)
    ]
    names = sorted(f.path for f in iter_files(build_observation_tree(obs)) if f.kind == "topic")
    assert names == ["misc-2026-w45.md", "misc-2026-w46.md"]


def _frontmatter(content: str) -> dict[str, str] | None:
    """The frontmatter's top-level ``key: value`` lines, or None when there is none."""
    if not content.startswith("---\n"):
        return None
    block = content[4 : content.index("\n---\n", 4)]
    top = [line.partition(":") for line in block.splitlines() if not line.startswith(" ")]
    return {k: v.strip() for k, _, v in top}


def test_the_tree_is_an_open_knowledge_format_bundle():
    """OKF v0.2 (https://okf.md/spec/): every concept file has frontmatter with a non-empty type,
    index.md is reserved for folder overviews and carries no frontmatter except the root's
    okf_version, timestamps carry a UTC offset, and links are bundle-relative and resolve."""
    files = list(iter_files(build_observation_tree(_realistic_bank())))
    paths = {f.path for f in files}
    for f in files:
        meta = _frontmatter(f.content)
        if f.kind == "index":
            assert f.name == "index.md"
            assert meta == ({"okf_version": f'"{OKF_VERSION}"'} if f.path == "index.md" else None), f.path
        else:
            assert f.name not in ("index.md", "log.md")
            assert meta and meta["type"], f.path
            # Strings and lists are written as JSON, which is valid YAML.
            assert json.loads(meta["title"]) == f.title
            assert json.loads(meta["description"])
            assert json.loads(meta["tags"]) == f.about
        for link in re.findall(r"\]\((/[^)]+)\)", f.content):
            assert link.lstrip("/") in paths, f"{f.path} links to missing {link}"


def test_frontmatter_timestamps_carry_an_offset():
    obs = [
        TreeObservation(
            id=f"o{i}",
            text=f"fact {i}",
            proof_count=1,
            entities=["Redis"],
            updated_at=datetime(2026, 10, i + 1, tzinfo=timezone.utc),
        )
        for i in range(3)
    ]
    meta = _frontmatter(next(f for f in iter_files(build_observation_tree(obs)) if f.kind == "topic").content)
    assert meta is not None and meta["timestamp"] == "2026-10-03T00:00:00+00:00"


def _subjects(names: list[str], per: int = LEAF_MAX) -> list[TreeObservation]:
    """``per`` observations about each named entity, plus a shared noise entity per pair so
    the bank is big enough to split."""
    return [
        TreeObservation(id=f"{n}-{i}", text=f"{n} fact {i}", proof_count=1, entities=[n, f"{n}-detail-{i % 5}"])
        for n in names
        for i in range(per)
    ]


def test_a_topic_never_takes_a_reserved_name():
    """An entity called "Index" must not produce a topic file that overwrites index.md."""
    paths = [f.path for f in iter_files(build_observation_tree(_subjects(["Index", "Log", "Misc", "Redis"])))]
    assert len(paths) == len(set(paths)), paths
    assert paths.count("index.md") == 1
    assert "log.md" not in paths


def test_non_latin_names_keep_their_letters():
    paths = {f.path for f in iter_files(build_observation_tree(_subjects(["北京", "東京", "서울", "Москва", "हिन्दी"])))}
    assert {"北京.md", "東京.md", "서울.md", "москва.md", "हिन्दी.md"} <= paths, paths


def test_every_path_is_lowercase():
    """Folder and file names are lowercase, including week files (misc-2026-w45.md), so a
    case-insensitive filesystem (macOS, Windows) cannot fold two files into one."""
    banks = [_realistic_bank(), _subjects(["Redis", "GPU", "Postgres", "Москва"])]
    weeks = [
        TreeObservation(
            id=f"m{i}",
            text=f"stray {i}",
            proof_count=1,
            entities=[f"oneoff-{i}"],
            mentioned_at=datetime(2026, 11, 2 + i % 14, tzinfo=timezone.utc),
        )
        for i in range(LEAF_MAX * 2)
    ]
    for obs in [*banks, weeks]:
        paths = [f.path for f in iter_files(build_observation_tree(obs))]
        assert all(p == p.lower() for p in paths), [p for p in paths if p != p.lower()]


def test_a_bank_small_enough_for_one_file_titles_its_topic_and_keeps_its_path():
    """With one topic nothing sets it apart from the bank: it is titled after its most common
    entities, but its path is fixed, so a new subject does not move every observation."""
    obs = [
        TreeObservation(id=f"{e}-{i}", text=f"{e} fact {i}", proof_count=1, entities=[e])
        for e in ["Kafka", "Nginx", "Postgres", "Redis"]
        for i in range(2)
    ]
    topics = [f for f in iter_files(build_observation_tree(obs)) if f.kind == "topic"]
    assert [(f.path, f.title) for f in topics] == [("all.md", "Kafka · Nginx")]
    more = obs + [
        TreeObservation(id=f"Aardvark-{i}", text=f"Aardvark fact {i}", proof_count=1, entities=["Aardvark"])
        for i in range(2)
    ]
    assert [f.path for f in iter_files(build_observation_tree(more)) if f.kind == "topic"] == ["all.md"]
