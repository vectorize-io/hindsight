"""Unit tests for in-batch new-entity name clustering (issue #3107).

Covers the pure halves of the in-batch dedup fix: `trigram_similarity` (an in-memory
reimplementation of Postgres pg_trgm) and `_cluster_new_entity_names` (union-find + canonical
selection). No DB, no LLM — deterministic.
"""

import math
import random

import pytest

from hindsight_api.engine.entity_resolver import (
    _cluster_new_entity_names,
    _find_intrabatch_similar_pairs,
    _SimilarNamePair,
    _trigram_set,
    _trigram_set_similarity,
    trigram_similarity,
)


# Expected values are what Postgres `SELECT similarity(lower(a), lower(b))` returns for each pair —
# this locks the in-memory implementation to pg_trgm so the calibrated 0.5 merge cutoff stays valid.
@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("Wren 🕯️", "Wren 🗯️", 1.0),  # emoji ignored → identical trigrams
        ("Aster", "Aster 🔑", 1.0),
        ("Aster", "aster 0", 0.75),  # case + numeric suffix
        ("Aster", "ke-aster", 2 / 3),
        ("Merrivale", "Merryvale", 7 / 13),  # typo ≈ 0.538
        ("Corvin", "Corvyn", 0.4),  # short typo — below cutoff
        ("Aster", "Astrid", 0.3),  # distinct person
        ("José García", "Jose Garcia", 7 / 17),  # accents
        ("北京", "北京市", 0.4),  # CJK
        ("Jean-Luc", "Jean Luc", 1.0),  # hyphen == space (both separators)
    ],
)
def test_trigram_similarity_matches_pg_trgm(a, b, expected):
    assert trigram_similarity(a, b) == pytest.approx(expected)


def test_trigram_similarity_is_symmetric_and_bounded():
    assert trigram_similarity("Corvin", "Corvyn") == trigram_similarity("Corvyn", "Corvin")
    assert trigram_similarity("", "") == 0.0
    assert trigram_similarity("Aster", "Aster") == 1.0


def test_find_pairs_applies_threshold():
    names = ["Aster", "aster 0", "Astrid"]
    # 0.5 cutoff: Aster~aster 0 (0.75) pairs; Aster~Astrid (0.30) does not.
    pairs = {(p.name_a, p.name_b) for p in _find_intrabatch_similar_pairs(names, 0.5)}
    assert pairs == {("Aster", "aster 0")}
    # A cutoff above every pair returns nothing.
    assert _find_intrabatch_similar_pairs(names, 0.99) == []


def _cluster(names, pairs, counts=None):
    rep = {n.lower(): n for n in names}
    count_by_lower = {n.lower(): (counts or {}).get(n, 1) for n in names}
    sim_pairs = [_SimilarNamePair(name_a=a, name_b=b) for a, b in pairs]
    cmap = _cluster_new_entity_names(rep, count_by_lower, sim_pairs)
    # Invert to canonical -> sorted members (by lowercase) for stable assertions.
    clusters: dict[str, list[str]] = {}
    for name_lower, canonical in cmap.items():
        clusters.setdefault(canonical, []).append(name_lower)
    return {canonical: sorted(members) for canonical, members in clusters.items()}


def test_singletons_map_to_themselves():
    assert _cluster(["Alice", "Bob"], pairs=[]) == {"Alice": ["alice"], "Bob": ["bob"]}


def test_transitive_pairs_form_one_cluster():
    # a~b and b~c must land all three in a single cluster even without a direct a~c pair.
    result = _cluster(["Aster", "aster 0", "Aster 🔑"], pairs=[("Aster", "aster 0"), ("aster 0", "Aster 🔑")])
    assert len(result) == 1
    assert sorted(next(iter(result.values()))) == ["aster", "aster 0", "aster 🔑"]
    assert list(result.keys()) == ["Aster"]  # shortest form is canonical


def test_canonical_prefers_most_mentioned():
    # "aster 0" is longer but mentioned more → it wins over the shorter "Aster".
    result = _cluster(["Aster", "aster 0"], pairs=[("Aster", "aster 0")], counts={"aster 0": 5, "Aster": 1})
    assert list(result.keys()) == ["aster 0"]


def test_canonical_prefers_shortest_when_counts_tie():
    result = _cluster(["Aster", "aster 0"], pairs=[("Aster", "aster 0")])
    assert list(result.keys()) == ["Aster"]


def test_canonical_lexicographic_tiebreak():
    # Same count and length → lexicographically smallest original spelling.
    result = _cluster(["abd", "abc"], pairs=[("abd", "abc")])
    assert list(result.keys()) == ["abc"]


def test_distinct_names_not_merged():
    # No pair between them → two clusters (mirrors "Aster"/"Astrid" staying apart).
    assert _cluster(["Aster", "Astrid"], pairs=[]) == {"Aster": ["aster"], "Astrid": ["astrid"]}


def test_pairs_are_case_insensitive():
    # The pg_trgm join lowercases; a pair reported in any case must still union.
    result = _cluster(["Wren 🕯️", "wren 🗯️"], pairs=[("WREN 🕯️", "Wren 🗯️")])
    assert len(result) == 1


def test_separate_clusters_stay_separate():
    result = _cluster(
        ["Wren 🕯️", "Wren 🗯️", "Merrivale", "Merryvale"],
        pairs=[("Wren 🕯️", "Wren 🗯️"), ("Merrivale", "Merryvale")],
    )
    assert len(result) == 2
    assert all(len(members) == 2 for members in result.values())


# --- The prefix-filtering join must return exactly what the quadratic loop returned -----------
#
# `_find_intrabatch_similar_pairs` prunes candidate pairs with a prefix filter instead of
# comparing all of them. The pruning is only allowed to skip pairs that could never clear the
# cutoff, so the join is pinned against the double loop it replaced rather than against a list
# of expected pairs — a bad prefix bound shows up as a *missing* pair, which no fixed example
# would catch.


def _pairs_by_exhaustive_comparison(names: list[str], threshold: float) -> set[tuple[str, str]]:
    """Every qualifying pair, found by comparing all of them — the pre-optimisation behaviour."""
    trigrams = [_trigram_set(n) for n in names]
    return {
        (names[i], names[j])
        for i in range(len(names))
        for j in range(i + 1, len(names))
        if _trigram_set_similarity(trigrams[i], trigrams[j]) >= threshold
    }


def _pairs_from_join(names: list[str], threshold: float) -> set[tuple[str, str]]:
    return {(p.name_a, p.name_b) for p in _find_intrabatch_similar_pairs(names, threshold)}


_WORD_PARTS = ["as", "ter", "wren", "mer", "ri", "vale", "cor", "vin", "北京", "josé", "ke", "0", "x"]


def _random_batch(rng: random.Random) -> list[str]:
    """A batch of distinct names built from shared fragments, so near-misses are common."""
    names: list[str] = []
    seen: set[str] = set()
    for _ in range(rng.randint(2, 40)):
        name = " ".join(rng.choice(_WORD_PARTS) for _ in range(rng.randint(1, 3)))
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


# Thresholds include values that are exactly achievable Jaccard ratios (1/3, 2/5, 7/13 ...): the
# prefix length is `|A| - ceil(t * |A|) + 1`, so a float `t * |A|` landing a hair above a whole
# number would shorten the prefix by one and silently drop a pair sitting exactly on the cutoff.
_EQUIVALENCE_THRESHOLDS = [0.1, 0.2, 1 / 3, 0.4, 2 / 5, 0.5, 7 / 13, 0.6, 2 / 3, 0.75, 0.8, 0.9, 0.99, 1.0]


@pytest.mark.parametrize("threshold", _EQUIVALENCE_THRESHOLDS)
def test_join_returns_exactly_the_exhaustive_pairs(threshold):
    rng = random.Random(20260907)
    for _ in range(40):
        names = _random_batch(rng)
        assert _pairs_from_join(names, threshold) == _pairs_by_exhaustive_comparison(names, threshold)


def test_join_keeps_a_pair_sitting_exactly_on_the_cutoff():
    # Merrivale/Merryvale is 7/13; at exactly that cutoff it is in, one ulp above it is out.
    names = ["Merrivale", "Merryvale"]
    assert _pairs_from_join(names, 7 / 13) == {("Merrivale", "Merryvale")}
    assert _pairs_from_join(names, math.nextafter(7 / 13, 1.0)) == set()


def test_join_emits_pairs_in_input_order():
    # The join visits names shortest-trigram-set-first, which is not input order; the pair it
    # emits must still name them the way the caller listed them.
    names = ["ke-aster", "Aster"]
    assert _pairs_from_join(names, 0.5) == {("ke-aster", "Aster")}


def test_join_ignores_names_with_no_trigrams():
    # "!!!" has no word characters, so pg_trgm gives it no trigrams and it can never reach a
    # positive cutoff — but it must not break the batch around it.
    names = ["!!!", "Aster", "aster 0", "***"]
    assert _pairs_from_join(names, 0.5) == {("Aster", "aster 0")}


def test_join_handles_a_batch_of_mutually_similar_names():
    # The shape that defeats prefix filtering: every name probes into every bucket. Correctness
    # must not depend on the pruning actually pruning anything.
    names = [f"Acme Corporation Subsidiary {i:04d}" for i in range(30)]
    assert _pairs_from_join(names, 0.5) == _pairs_by_exhaustive_comparison(names, 0.5)
