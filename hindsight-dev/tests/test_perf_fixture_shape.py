"""The perf fixture's entity-graph shape (issue #3510).

``recall-with-observations`` only exercises the observation graph arm's real
failure mode if its synthetic bank is shaped like a real one. The property that
does the work is the *post-resolution* entity graph: a few hubs over a long tail
of rarely-mentioned entities, so a handful of observation seeds reaches a small
fraction of the graph rather than all of it.

These tests assert that outcome rather than the mechanism. The mechanism has
already failed in a non-obvious way once — tail names generated as
"<stem> <counter>" scored 0.73 pg_trgm against their neighbours, so the entity
resolver merged 2,814 generated names back down to 159 and the bank came out with
the flat hub-only graph the vocabulary exists to avoid. Checking name distances
directly would flag harmless pairs and still miss that outcome.
"""

import statistics
from collections import Counter

from benchmarks.perf.recall_perf import (
    ENTITIES,
    FACT_TEMPLATES,
    _build_vocabulary,
    _fill_template,
    configure_entity_vocabulary,
)

# The entity resolver fuzzy-merges two names within a retain batch at or above
# this pg_trgm similarity (EntityResolver.intrabatch_merge_similarity).
_MERGE_SIMILARITY = 0.5
_CORPUS = 5_000


def _mentions_after_resolution(corpus_size: int) -> Counter:
    """Entity mention counts as the bank would hold them, post fuzzy-merge.

    Simulates the resolver's intra-batch merge so the shape assertions below
    describe the graph that actually lands in ``unit_entities``, not the names
    the generator emitted.
    """
    from hindsight_api.engine.entity_resolver import _trigram_similarity

    configure_entity_vocabulary(corpus_size)
    raw: Counter = Counter()
    for i in range(corpus_size):
        for entity in _fill_template(FACT_TEMPLATES[i % len(FACT_TEMPLATES)]).entities[:3]:
            raw[entity] += 1

    canonical: dict[str, str] = {}
    for name in raw:
        for seen in canonical:
            if _trigram_similarity(name, seen) >= _MERGE_SIMILARITY:
                canonical[name] = canonical[seen]
                break
        else:
            canonical[name] = name

    merged: Counter = Counter()
    for name, count in raw.items():
        merged[canonical[name]] += count
    return merged


def test_vocabulary_grows_with_the_corpus():
    """A fixed-size vocabulary makes degree scale with bank size, not entity count."""
    assert len(_build_vocabulary(len(ENTITIES))) == len(ENTITIES), "the hub head is the floor"
    assert len(_build_vocabulary(500)) < len(_build_vocabulary(5_000)), "vocabulary must grow with the corpus"


def test_entity_graph_keeps_a_long_tail():
    """Hubs over a tail of rare entities — the shape the #3510 bank had.

    That bank ran median degree 2 with a third of its entities mentioned once and
    its top entity on 8.7% of links. The old fixed list produced median degree 40
    and *zero* entities mentioned once, so every seed reached the whole graph.
    """
    merged = _mentions_after_resolution(_CORPUS)
    total = sum(merged.values())

    assert len(merged) > 800, (
        f"only {len(merged)} entities survived resolution for {_CORPUS} facts — the tail is "
        "collapsing, most likely because generated names fuzzy-merge into each other"
    )
    assert statistics.median(merged.values()) <= 8, "the median entity must stay rare"
    assert sum(1 for c in merged.values() if c == 1) > 50, "a real bank has entities seen once"
    assert max(merged.values()) / total > 0.02, "and it still has hubs"
