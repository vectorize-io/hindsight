from unittest.mock import MagicMock

from hindsight_api.engine.retain.fact_extraction import _build_extraction_prompt_and_schema


def _config(mode: str) -> MagicMock:
    config = MagicMock()
    config.entity_labels = None
    config.entities_allow_free_form = True
    config.retain_extraction_mode = mode
    config.retain_extract_causal_links = False
    config.retain_mission = None
    config.retain_custom_instructions = "Extract project decisions" if mode == "custom" else None
    config.llm_output_language = None
    return config


def test_concise_examples_are_explicitly_excluded_from_extracted_content():
    prompt, _ = _build_extraction_prompt_and_schema(_config("concise"))

    guard = (
        "The examples below demonstrate output format and selectivity only. Never emit\n"
        "their facts, entities, or dates unless those details also appear in the actual\n"
        "input text being processed."
    )
    assert guard in prompt
    assert prompt.index(guard) < prompt.index("Example 1 - Selective extraction")


def test_custom_mode_does_not_include_examples_or_their_guard():
    prompt, _ = _build_extraction_prompt_and_schema(_config("custom"))

    assert "Example 1 - Selective extraction" not in prompt
    assert "The examples below demonstrate output format" not in prompt
