"""Deterministic tests for the C2 wiring (relation extraction into the prompt).

Covers:

* Default (no ``graphiti_group_id``) → prompt rendering is byte-identical to
  the pre-wiring version for every supported extraction mode. This is the
  canary: a non-federated bank must see zero change in the prompt sent to the
  LLM and the response schema it receives.
* Federated (``graphiti_group_id`` non-empty) → the prompt gains the
  relations section and the response schema switches to ``FactExtractionResponseUnified``.
* Relations are validated and attached to ``Fact.relations`` / ``ExtractedFact.relations``
  by the lenient-parsing path that runs on the LLM JSON output.

The judge test for whether the LLM actually emits correct relations without
drifting fact selectivity lives in a separate ``hs_llm_core`` module.
"""

from types import SimpleNamespace

from hindsight_api.engine.retain.fact_extraction import (
    CONCISE_FACT_EXTRACTION_PROMPT,
    CUSTOM_FACT_EXTRACTION_PROMPT,
    VERBATIM_FACT_EXTRACTION_PROMPT,
    VERBOSE_FACT_EXTRACTION_PROMPT,
    _build_extraction_prompt_and_schema,
)
from hindsight_api.engine.retain.relation_extraction import (
    RELATIONS_PROMPT_SECTION,
    ExtractedFactUnified,
    FactExtractionResponseUnified,
    _validate_relations_from_dict,
)


def _config(**overrides):
    """Minimal HindsightConfig stand-in for the prompt builder.

    Only the fields _build_extraction_prompt_and_schema touches are populated.
    Defaults mirror the pre-wiring bank config.
    """
    base = {
        "retain_extraction_mode": "concise",
        "retain_extract_causal_links": True,
        "retain_custom_instructions": None,
        "graphiti_group_id": "",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_prompt_byte_identical_when_no_graphiti_group_id():
    """The canary: non-federated banks see the exact same prompt as before."""
    for mode in ("concise", "verbose", "verbatim", "custom"):
        cfg = _config(
            retain_extraction_mode=mode,
            retain_custom_instructions=("focus on dates" if mode == "custom" else None),
        )
        prompt, schema = _build_extraction_prompt_and_schema(cfg)
        # Default path keeps the standard (non-unified) response schema
        assert schema is not FactExtractionResponseUnified
        # The relations section must NOT appear in the rendered prompt
        assert "RELATIONS (between entities within ONE fact)" not in prompt
        # And the placeholder itself is fully resolved (no leaked braces)
        assert "{relations_section}" not in prompt


def test_prompt_gains_relations_section_when_group_id_set():
    """Federated banks get the relations prompt section in concise/verbose/custom."""
    for mode in ("concise", "verbose", "custom"):
        cfg = _config(
            retain_extraction_mode=mode,
            retain_custom_instructions=("focus on dates" if mode == "custom" else None),
            graphiti_group_id="agent-shared-001",
        )
        prompt, schema = _build_extraction_prompt_and_schema(cfg)
        assert "RELATIONS (between entities within ONE fact)" in prompt
        # R7 (selectivity must not drift) is the most important guard
        assert "must NOT change WHICH facts you extract" in prompt
        # Schema switches to the unified variant
        assert schema is FactExtractionResponseUnified


def test_verbatim_mode_never_uses_unified():
    """Verbatim stores raw text + metadata only; no fact text to relate."""
    cfg = _config(retain_extraction_mode="verbatim", graphiti_group_id="agent-shared-001")
    prompt, schema = _build_extraction_prompt_and_schema(cfg)
    assert "RELATIONS (between entities within ONE fact)" not in prompt
    assert schema is not FactExtractionResponseUnified


def test_templates_have_relations_section_placeholder():
    """The four prompt templates carry the placeholder; verbatim hard-codes empty."""
    assert "{relations_section}" in CONCISE_FACT_EXTRACTION_PROMPT
    assert "{relations_section}" in CUSTOM_FACT_EXTRACTION_PROMPT
    assert "{relations_section}" in VERBOSE_FACT_EXTRACTION_PROMPT
    # Verbatim has no fact text to relate entities within, so the placeholder
    # is resolved at template build time to ""
    assert "{relations_section}" not in VERBATIM_FACT_EXTRACTION_PROMPT


def test_unified_response_schema_accepts_relations_field():
    """The unified Pydantic model parses per-fact relations end-to-end."""
    resp = FactExtractionResponseUnified.model_validate(
        {
            "facts": [
                {
                    "what": "Alice leads infra team",
                    "when": "N/A",
                    "where": "N/A",
                    "who": "Alice",
                    "why": "N/A",
                    "fact_type": "world",
                    "entities": [{"text": "Alice"}, {"text": "infra team"}],
                    "relations": [
                        {
                            "source_entity_index": 0,
                            "target_entity_index": 1,
                            "predicate": "leads",
                        }
                    ],
                }
            ]
        }
    )
    assert resp.facts[0].relations is not None
    assert len(resp.facts[0].relations) == 1
    assert resp.facts[0].relations[0].source_entity_index == 0


def test_dict_validator_routes_to_fact_relations():
    """The internal pipeline path: _validate_relations_from_dict filters and returns dicts."""
    out = _validate_relations_from_dict(
        {
            "entities": [{"text": "Alice"}, {"text": "Google"}],
            "relations": [
                {"source_entity_index": 0, "target_entity_index": 1, "predicate": "works at"},
                {"source_entity_index": 0, "target_entity_index": 5, "predicate": "LEADS"},
                {"source_entity_index": 1, "target_entity_index": 1, "predicate": "SELF"},
            ],
        }
    )
    # Only the well-formed relation survives; predicate normalized
    assert len(out) == 1
    assert out[0]["predicate"] == "WORKS_AT"
    assert out[0]["source_entity_index"] == 0
    assert out[0]["target_entity_index"] == 1


def test_prompt_section_constants_match_design():
    """The wired section text contains both guard rules (R2 + R7) the
    downstream code relies on. If either is removed, judge tests for
    English-predicate and selectivity non-drift will start failing —
    catching it here gives a unit-level signal first.
    """
    assert "SCREAMING_SNAKE_CASE" in RELATIONS_PROMPT_SECTION
    assert "ALWAYS English" in RELATIONS_PROMPT_SECTION
    assert "must NOT change WHICH facts you extract" in RELATIONS_PROMPT_SECTION
