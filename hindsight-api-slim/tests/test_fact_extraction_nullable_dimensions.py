"""The four descriptive extraction fields must accept null (#4457).

`when` / `where` / `who` / `why` used to be required non-null strings. Under
strict structured output every property is required, so a grammar-constrained
model had no legal way to say "the text doesn't state this for this fact" — and
a small local model asked to produce *some* string reaches for the nearest
plausible one, which is whatever the surrounding text mentions. The reported
case copied an unrelated project date into a fact's `when`; a fabricated date
becomes durable memory and gets acted on later.

The fix keeps the keys required (so the model can't silently drop them) and
makes the *values* nullable, exactly like `occurred_start` / `occurred_end`
already were. These tests pin both halves of that, across every fact model the
prompt builder can pick, plus the parse-side contract that null and the legacy
"N/A" placeholder both mean "absent" and stay out of the stored fact text.
"""

from unittest.mock import MagicMock

import pytest

from hindsight_api.engine.retain.fact_extraction import _build_extraction_prompt_and_schema
from hindsight_api.engine.structured_output import strict_json_schema

# (retain_extraction_mode, retain_extract_causal_links) -> every fact model the
# builder can pick. Verbatim has no `what`/`why`; the rest carry all four.
EXTRACTION_MODES = (
    ("concise", True),
    ("concise", False),
    ("verbose", True),
    ("verbatim", False),
)

DESCRIPTIVE_FIELDS = ("when", "where", "who", "why")


def _config(*, mode: str, causal: bool) -> MagicMock:
    config = MagicMock()
    config.retain_extraction_mode = mode
    config.retain_extract_causal_links = causal
    config.retain_custom_instructions = None
    config.retain_mission = None
    config.entity_labels = None
    config.entities_allow_free_form = True
    config.llm_output_language = None
    config.llm_supports_string_pattern = False
    return config


def _fact_model(mode: str, causal: bool) -> type:
    """The per-fact model inside the response wrapper the builder returns."""
    _, response_schema = _build_extraction_prompt_and_schema(_config(mode=mode, causal=causal))
    return response_schema.model_fields["facts"].annotation.__args__[0]


def _fact_definition(mode: str, causal: bool) -> dict:
    """The strict-subset schema for the per-fact object, as a provider sees it."""
    _, response_schema = _build_extraction_prompt_and_schema(_config(mode=mode, causal=causal))
    definitions = strict_json_schema(response_schema)["$defs"].values()
    return next(d for d in definitions if "when" in d.get("properties", {}))


@pytest.mark.parametrize(("mode", "causal"), EXTRACTION_MODES)
@pytest.mark.parametrize("field", DESCRIPTIVE_FIELDS)
def test_strict_schema_keeps_the_key_required(mode, causal, field):
    """Required, so a model can't quietly drop the dimension entirely."""
    definition = _fact_definition(mode, causal)
    if field not in definition["properties"]:
        pytest.skip(f"{mode} mode has no '{field}' field")

    assert field in definition["required"]


@pytest.mark.parametrize(("mode", "causal"), EXTRACTION_MODES)
@pytest.mark.parametrize("field", DESCRIPTIVE_FIELDS)
def test_strict_schema_accepts_string_or_null(mode, causal, field):
    """Nullable, so "not stated here" is a legal answer and not a dare to invent one."""
    definition = _fact_definition(mode, causal)
    if field not in definition["properties"]:
        pytest.skip(f"{mode} mode has no '{field}' field")
    branches = definition["properties"][field].get("anyOf", [])

    assert {"type": "string"} in branches
    assert {"type": "null"} in branches
    # OpenAI strict rejects `default`; the generator strips it. Pin that it stayed stripped.
    assert "default" not in definition["properties"][field]


@pytest.mark.parametrize(("mode", "causal"), EXTRACTION_MODES)
@pytest.mark.parametrize("field", DESCRIPTIVE_FIELDS)
def test_model_validates_null(mode, causal, field):
    model = _fact_model(mode, causal)
    if field not in model.model_fields:
        pytest.skip(f"{mode} mode has no '{field}' field")
    payload = {"fact_type": "world", "what": "Customer notifications precede fieldwork", field: None}

    assert getattr(model.model_validate(payload), field) is None


@pytest.mark.parametrize(("mode", "causal"), EXTRACTION_MODES)
def test_descriptions_stop_asking_for_the_n_a_placeholder(mode, causal):
    """The schema descriptions are the model's only instructions per field.

    Leaving "write 'N/A'" in them re-creates the same pressure the nullable type
    removes: a model told to write a placeholder string still has to write a
    string, and "N/A" competes with the nearby date for that slot.
    """
    model = _fact_model(mode, causal)

    for field in DESCRIPTIVE_FIELDS:
        description = (model.model_fields[field].description or "") if field in model.model_fields else ""
        assert "N/A" not in description, f"{field} still tells the model to write 'N/A'"
        if description:
            assert "null" in description
