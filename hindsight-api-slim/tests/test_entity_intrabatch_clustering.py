"""Unit tests for in-batch new-entity name clustering (issue #3107).

`_cluster_new_entity_names` is the pure union-find + canonical-selection half of the in-batch
dedup fix (the pg_trgm self-join that produces the pairs is covered by the integration test).
No DB, no LLM — deterministic given (names, mention counts, similar pairs).
"""

from hindsight_api.engine.entity_resolver import _cluster_new_entity_names, _SimilarNamePair


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
