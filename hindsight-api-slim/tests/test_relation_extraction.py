"""Deterministic tests for the relation-extraction prototype (federation mode).

Covers the structural mechanics only — schema shape, defensive validation,
predicate normalization. The model-following behaviour (does the LLM emit
correct relations without changing fact selectivity) belongs in a separate
``hs_llm_core`` judge test once the mode is wired into the extraction
pipeline.
"""

from hindsight_api.engine.retain.fact_extraction import Entity
from hindsight_api.engine.retain.relation_extraction import (
    RELATIONS_PROMPT_SECTION,
    ExtractedFactUnified,
    ExtractedRelation,
    FactExtractionResponseUnified,
    normalize_predicate,
    validate_relations,
)


def _fact(entities: list[str], relations: list[ExtractedRelation] | None) -> ExtractedFactUnified:
    return ExtractedFactUnified(
        what="Alice leads infrastructure team",
        when="N/A",
        where="N/A",
        who="Alice",
        why="N/A",
        fact_type="world",
        entities=[Entity(text=e) for e in entities],
        relations=relations,
    )


def test_valid_relation_passes_through():
    rel = ExtractedRelation(source_entity_index=0, target_entity_index=1, predicate="LEADS")
    result = validate_relations(_fact(["Alice", "infrastructure team"], [rel]))
    assert len(result) == 1
    assert result[0].predicate == "LEADS"


def test_out_of_range_index_dropped():
    rels = [
        ExtractedRelation(source_entity_index=0, target_entity_index=5, predicate="LEADS"),
        ExtractedRelation(source_entity_index=-1, target_entity_index=0, predicate="LEADS"),
    ]
    assert validate_relations(_fact(["Alice", "team"], rels)) == []


def test_self_relation_dropped():
    rel = ExtractedRelation(source_entity_index=1, target_entity_index=1, predicate="KNOWS")
    assert validate_relations(_fact(["Alice", "Bob"], [rel])) == []


def test_invalid_predicate_dropped_and_sloppy_predicate_normalized():
    bad = ExtractedRelation(source_entity_index=0, target_entity_index=1, predicate="works at!")
    sloppy = ExtractedRelation(source_entity_index=0, target_entity_index=1, predicate="works at")
    result = validate_relations(_fact(["Alice", "Google"], [bad, sloppy]))
    assert len(result) == 1
    assert result[0].predicate == "WORKS_AT"


def test_no_relations_is_normal():
    assert validate_relations(_fact(["Alice"], None)) == []
    assert validate_relations(_fact(["Alice"], [])) == []


def test_no_entities_drops_all_relations():
    rel = ExtractedRelation(source_entity_index=0, target_entity_index=1, predicate="LEADS")
    fact = _fact([], [rel])
    assert validate_relations(fact) == []


def test_normalize_predicate_variants():
    assert normalize_predicate("works at") == "WORKS_AT"
    assert normalize_predicate("lives-in") == "LIVES_IN"
    assert normalize_predicate(" MARRIED_TO ") == "MARRIED_TO"


def test_temporal_fields_optional_and_preserved():
    rel = ExtractedRelation(
        source_entity_index=0,
        target_entity_index=1,
        predicate="WORKS_AT",
        rel_valid_at="2024-06-10T00:00:00Z",
    )
    result = validate_relations(_fact(["Bob", "Google"], [rel]))
    assert result[0].rel_valid_at == "2024-06-10T00:00:00Z"
    assert result[0].rel_invalid_at is None


def test_unified_response_schema_parses_plain_facts():
    # Facts without relations (non-federated output shape) parse unchanged.
    resp = FactExtractionResponseUnified.model_validate(
        {
            "facts": [
                {
                    "what": "Alice joined Google",
                    "when": "N/A",
                    "where": "N/A",
                    "who": "Alice",
                    "why": "N/A",
                    "fact_type": "world",
                }
            ]
        }
    )
    assert resp.facts[0].relations is None


def test_prompt_section_states_critical_rules():
    # The two rules that guard against regressions elsewhere in the pipeline:
    # selectivity must not drift (R7) and predicates stay English (R2).
    assert "must NOT change WHICH facts you extract" in RELATIONS_PROMPT_SECTION
    assert "SCREAMING_SNAKE_CASE" in RELATIONS_PROMPT_SECTION
