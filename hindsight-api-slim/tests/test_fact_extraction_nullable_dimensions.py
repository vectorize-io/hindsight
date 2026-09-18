"""`retain_optional_fact_dimensions` — the opt-in that lets a fact say "not stated" (#4457).

Under strict structured output every declared property is required, so a model
asked for `when` on a fact the text gives no date for has no legal way to say
"not stated": it must emit a string, and the nearest plausible one is whatever
the surrounding text mentions. A project's start date became the deadline of an
unrelated requirement, and a fabricated date is durable memory nothing
downstream can tell from a real one.

The flag makes the four descriptive values nullable. It is **off by default**,
and that is the half worth pinning: turning it on is not neutral even on a
capable model. With `why` droppable, "the user asked me to refactor X" comes
back as its own `world` fact instead of riding along as the agent fact's
rationale — a defensible reading, but a different one, which is why existing
deployments keep today's behaviour until an operator opts in.

So these tests assert both states: off is byte-for-byte the old contract, on is
nullable-but-still-required. Everything here is structural and deterministic —
whether a real model actually stops borrowing the date is judged in
`test_extraction_absent_dimensions.py`.
"""

from unittest.mock import MagicMock

import pytest

from hindsight_api.config import DEFAULT_RETAIN_OPTIONAL_FACT_DIMENSIONS
from hindsight_api.engine.retain.fact_extraction import _build_extraction_prompt_and_schema
from hindsight_api.engine.structured_output import strict_json_schema

# (retain_extraction_mode, retain_extract_causal_links) -> every fact model the
# builder can pick, so the flag cannot be wired into just one of them.
EXTRACTION_MODES = (
    ("concise", True),
    ("concise", False),
    ("verbose", True),
    ("verbatim", False),
)

DESCRIPTIVE_FIELDS = ("when", "where", "who", "why")


def _config(*, mode: str, causal: bool, optional_dimensions: bool) -> MagicMock:
    config = MagicMock()
    config.retain_extraction_mode = mode
    config.retain_extract_causal_links = causal
    config.retain_optional_fact_dimensions = optional_dimensions
    config.retain_custom_instructions = None
    config.retain_mission = None
    config.entity_labels = None
    config.entities_allow_free_form = True
    config.llm_output_language = None
    config.llm_supports_string_pattern = False
    return config


def _fact_definition(mode: str, causal: bool, optional_dimensions: bool) -> dict:
    """The strict-subset schema for the per-fact object, as a provider sees it."""
    _, response_schema = _build_extraction_prompt_and_schema(
        _config(mode=mode, causal=causal, optional_dimensions=optional_dimensions)
    )
    definitions = strict_json_schema(response_schema)["$defs"].values()
    return next(d for d in definitions if "when" in d.get("properties", {}))


def _prompt(mode: str, causal: bool, optional_dimensions: bool) -> str:
    prompt, _ = _build_extraction_prompt_and_schema(
        _config(mode=mode, causal=causal, optional_dimensions=optional_dimensions)
    )
    return prompt


def test_the_flag_is_off_by_default():
    """Turning it on changes what a capable model returns, so nobody gets it by surprise."""
    assert DEFAULT_RETAIN_OPTIONAL_FACT_DIMENSIONS is False


@pytest.mark.parametrize(("mode", "causal"), EXTRACTION_MODES)
@pytest.mark.parametrize("field", DESCRIPTIVE_FIELDS)
def test_off_keeps_the_required_non_null_contract(mode, causal, field):
    """The default path must serialize exactly as it did before the flag existed."""
    definition = _fact_definition(mode, causal, optional_dimensions=False)
    if field not in definition["properties"]:
        pytest.skip(f"{mode} mode has no '{field}' field")
    schema = definition["properties"][field]

    assert schema.get("type") == "string"
    assert "anyOf" not in schema
    assert field in definition["required"]


@pytest.mark.parametrize(("mode", "causal"), EXTRACTION_MODES)
@pytest.mark.parametrize("field", DESCRIPTIVE_FIELDS)
def test_on_makes_the_value_nullable_but_keeps_the_key_required(mode, causal, field):
    """Nullable, so "not stated" is a legal answer; required, so the key can't vanish."""
    definition = _fact_definition(mode, causal, optional_dimensions=True)
    if field not in definition["properties"]:
        pytest.skip(f"{mode} mode has no '{field}' field")
    schema = definition["properties"][field]

    assert {"type": "string"} in schema["anyOf"]
    assert {"type": "null"} in schema["anyOf"]
    assert field in definition["required"]
    # OpenAI strict rejects `default`; the generator strips it. Pin that it stayed stripped.
    assert "default" not in schema


@pytest.mark.parametrize(("mode", "causal"), EXTRACTION_MODES)
def test_on_stops_asking_for_the_n_a_placeholder(mode, causal):
    """Schema and prompt have to agree, or the instructions fight the grammar.

    Only the placeholder is swapped, nothing else is reworded. Rewriting these
    descriptions properly ("explicitly stated for THIS fact … never invent a
    motive") is what flipped the experience/world balance in
    test_fact_extraction_agent_experience, so the substitution stays mechanical.
    """
    definition = _fact_definition(mode, causal, optional_dimensions=True)
    prompt = _prompt(mode, causal, optional_dimensions=True)

    for field in DESCRIPTIVE_FIELDS:
        description = definition["properties"].get(field, {}).get("description", "")
        assert "N/A" not in description, f"{field} still tells the model to write 'N/A'"
    # Everything before the opt-in section must be clean. The section itself names
    # the placeholder on purpose — "write null, not N/A" is the instruction — so
    # asserting over the whole prompt would forbid the very sentence doing the work.
    body, _, optional_section = prompt.partition("OPTIONAL FIELDS")
    assert "N/A" not in body, "the FACT FORMAT block still asks for the placeholder"
    assert 'not "N/A"' in optional_section


@pytest.mark.parametrize(("mode", "causal"), EXTRACTION_MODES)
def test_off_leaves_the_prompt_untouched(mode, causal):
    """The opt-in section is the only prompt difference, and it is opt-in."""
    assert "OPTIONAL FIELDS" not in _prompt(mode, causal, optional_dimensions=False)
    assert "OPTIONAL FIELDS" in _prompt(mode, causal, optional_dimensions=True)
