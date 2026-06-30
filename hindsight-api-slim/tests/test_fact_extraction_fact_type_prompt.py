from unittest.mock import MagicMock

from hindsight_api.engine.retain.fact_extraction import (
    ExtractedFact,
    ExtractedFactNoCausal,
    ExtractedFactVerbose,
    _build_extraction_prompt_and_schema,
)


def _baseline_config() -> MagicMock:
    config = MagicMock()
    config.entity_labels = None
    config.entities_allow_free_form = True
    config.retain_extraction_mode = "concise"
    config.retain_extract_causal_links = False
    config.retain_mission = None
    config.retain_custom_instructions = None
    config.llm_output_language = None
    return config


def test_concise_prompt_keeps_user_preferences_rules_and_corrections_world():
    prompt, _ = _build_extraction_prompt_and_schema(_baseline_config())

    assert '"world": Objective/external facts' in prompt
    assert "user's preferences, rules, corrections, constraints" in prompt
    assert 'These stay "world" even when the user states them during an assistant interaction' in prompt
    assert "Use this for the assistant/agent doing" in prompt
    assert "not merely for user facts mentioned in conversation" in prompt


def test_fact_type_schema_descriptions_distinguish_user_facts_from_agent_actions():
    for model in (ExtractedFact, ExtractedFactVerbose, ExtractedFactNoCausal):
        description = model.model_fields["fact_type"].description

        assert description is not None
        assert "preferences" in description
        assert "rules" in description
        assert "corrections" in description
        assert "assistant/agent actually performed" in description
