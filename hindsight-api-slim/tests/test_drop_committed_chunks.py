"""Dropping already-stored chunks from a retain batch keeps every positional reference right (#6337).

A batch keys phase-1 entity resolution and causal relations by fact position. When the chunks a
concurrent retain of the same content already stored are dropped, the kept facts move, so every
reference has to follow them, and the mentions of the dropped facts must not be counted.
"""

from hindsight_api.engine.retain.orchestrator import _drop_committed_chunks
from hindsight_api.engine.retain.types import (
    CausalRelation,
    ChunkMetadata,
    EntityResolutionResult,
    ExtractedFact,
    Phase1Result,
    ProcessedFact,
    ResolvedEntity,
)


class _Resolver:
    def __init__(self) -> None:
        self.forgotten: list[str] = []

    def forget_pending_mentions(self, entity_ids: list[str]) -> None:
        self.forgotten.extend(entity_ids)


def _batch():
    # Chunks 0, 1, 2 with one fact each, plus a second fact in chunk 2 caused by its first.
    extracted = [
        ExtractedFact(fact_text="a", fact_type="world", chunk_index=0),
        ExtractedFact(fact_text="b", fact_type="world", chunk_index=1),
        ExtractedFact(fact_text="c", fact_type="world", chunk_index=2),
        ExtractedFact(
            fact_text="d",
            fact_type="world",
            chunk_index=2,
            causal_relations=[CausalRelation(relation_type="caused_by", target_fact_index=2)],
        ),
    ]
    processed = [ProcessedFact.from_extracted_fact(f, [0.0, 1.0]) for f in extracted]
    chunks = [ChunkMetadata(chunk_text=t, fact_count=1, content_index=i, chunk_index=i) for i, t in enumerate("xyz")]
    ents = [ResolvedEntity(entity_id=f"e{i}", canonical_name=f"E{i}", entity_kind="regular") for i in range(4)]
    phase1 = Phase1Result(
        entities=EntityResolutionResult(
            resolved_entities=ents,
            entity_to_unit=[(str(i), i, None) for i in range(4)],
            unit_to_entity_ids={str(i): [f"e{i}"] for i in range(4)},
        ),
        semantic_ann_links=[("1", "old-unit", "semantic", 0.9, None), ("3", "old-unit", "semantic", 0.8, None)],
    )
    return chunks, extracted, processed, phase1


def test_dropping_a_chunk_renumbers_everything_that_points_at_a_fact():
    chunks, extracted, processed, phase1 = _batch()
    resolver = _Resolver()

    result = _drop_committed_chunks({1}, chunks, extracted, processed, phase1, resolver)

    assert [c.chunk_index for c in chunks] == [0, 2]
    assert [f.fact_text for f in extracted] == ["a", "c", "d"]
    assert [f.fact_text for f in processed] == ["a", "c", "d"]
    # "d" was caused by "c", which moved from position 2 to 1.
    for facts in (extracted, processed):
        assert [r.target_fact_index for r in facts[2].causal_relations] == [1]
    assert [e.entity_id for e in result.entities.resolved_entities] == ["e0", "e2", "e3"]
    assert [link[0] for link in result.entities.entity_to_unit] == ["0", "1", "2"]
    assert result.entities.unit_to_entity_ids == {"0": ["e0"], "1": ["e2"], "2": ["e3"]}
    assert [link[0] for link in result.semantic_ann_links] == ["2"]
    assert resolver.forgotten == ["e1"]


def test_a_relation_to_a_dropped_fact_is_removed_not_redirected():
    chunks, extracted, processed, phase1 = _batch()

    _drop_committed_chunks({2}, chunks, extracted, processed, phase1, _Resolver())

    assert [f.fact_text for f in extracted] == ["a", "b"]
    assert all(not f.causal_relations for f in extracted + processed)


def test_dropping_every_chunk_leaves_an_empty_batch():
    chunks, extracted, processed, phase1 = _batch()
    resolver = _Resolver()

    result = _drop_committed_chunks({0, 1, 2}, chunks, extracted, processed, phase1, resolver)

    assert chunks == [] and extracted == [] and processed == []
    assert result.entities.resolved_entities == [] and result.semantic_ann_links == []
    assert sorted(resolver.forgotten) == ["e0", "e1", "e2", "e3"]
