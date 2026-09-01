"""The `occurred_start` / `occurred_end` extraction contract.

Both fields are LLM-facing and typed `str | None`, so the JSON schema advertises
"any string". Under grammar-constrained decoding (`response_format: json_schema`
-> GBNF) the grammar is the only thing standing between the model and an
arbitrary string -- a description is not a constraint. When the model has to
emit a timestamp it cannot derive -- e.g. a narrative duration with no start
time -- it reasons *inside the string* and runs to the completion cap:

    "occurred_start": "2026-08-20T00:00:00Z/N/A (duration 40 mins implied ...
     Actually, I will set occurred_start/end to ... Wait, the prompt says "

which is a truncated body (`finish_reason: "length"`) that the retain path then
re-sends byte-identical (#3811, #3683).

A JSON Schema `pattern` makes that structurally impossible, but only on backends
that accept the keyword: Bedrock validates schemas against an allowlist that
excludes it and 400s the request outright (the same reason
`llm_supports_max_items` exists), and OpenAI errors on unsupported keywords under
`strict`. So it is opt-in via HINDSIGHT_API_LLM_SUPPORTS_STRING_PATTERN and
layered on at schema-build time -- these tests pin both halves: off by default,
correct when enabled.
"""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from hindsight_api.config import DEFAULT_LLM_SUPPORTS_STRING_PATTERN
from hindsight_api.engine.retain.fact_extraction import (
    Fact,
    _build_extraction_prompt_and_schema,
)

OCCURRED_FIELDS = ("occurred_start", "occurred_end")

# (retain_extraction_mode, retain_extract_causal_links) -> every fact model the
# builder can pick, so the constraint can't be wired into just one of them.
EXTRACTION_MODES = (
    ("concise", True),
    ("concise", False),
    ("verbose", True),
    ("verbatim", False),
)

# Formats real extraction models emit, all of which must keep working.
VALID_TIMESTAMPS = (
    "2026-08-20",
    "2026-08-20T00:00:00Z",
    "2026-08-19T00:00:00",
    "2026-08-20T00:00:00.000Z",
    "2026-08-20T14:30",
    "2026-08-20T14:30:00+02:00",
    "2026-08-20 14:30:00",
)

# Prose the unconstrained schema accepts. The last one is the observed runaway.
INVALID_TIMESTAMPS = (
    "N/A",
    "ongoing",
    "before Friday",
    "Starting on August 20, 2026",
    "2026-08-20T00:00:00Z/N/A (duration 40 mins implied ... Wait, the prompt says ",
)


def _config(*, mode: str, causal: bool, supports_pattern: bool) -> MagicMock:
    config = MagicMock()
    config.retain_extraction_mode = mode
    config.retain_extract_causal_links = causal
    config.retain_custom_instructions = None
    config.retain_mission = None
    config.entity_labels = None
    config.entities_allow_free_form = True
    config.llm_output_language = None
    config.llm_supports_string_pattern = supports_pattern
    return config


def _fact_model(config: MagicMock) -> type:
    """The per-fact model inside the response wrapper the builder returns."""
    _, response_schema = _build_extraction_prompt_and_schema(config)
    return response_schema.model_fields["facts"].annotation.__args__[0]


def _minimal(model: type) -> dict:
    """Smallest payload satisfying this model's required fields."""
    stubs = {"fact_type": "world", "what": "Marcus deployed an untested change"}
    return {name: stubs.get(name, "N/A") for name in model.model_json_schema().get("required", [])}


def _string_branch(model: type, field: str) -> dict:
    schema = model.model_json_schema()["properties"][field]
    return next(b for b in schema.get("anyOf", [schema]) if b.get("type") == "string")


def test_pattern_is_opt_in():
    """Narrow provider support and a hard 400 on rejection -- so, off by default."""
    assert DEFAULT_LLM_SUPPORTS_STRING_PATTERN is False


@pytest.mark.parametrize(("mode", "causal"), EXTRACTION_MODES)
@pytest.mark.parametrize("field", OCCURRED_FIELDS)
def test_schema_carries_no_pattern_when_unsupported(mode, causal, field):
    """The default schema must stay inside every backend's keyword subset."""
    model = _fact_model(_config(mode=mode, causal=causal, supports_pattern=False))

    assert "pattern" not in _string_branch(model, field)


@pytest.mark.parametrize(("mode", "causal"), EXTRACTION_MODES)
@pytest.mark.parametrize("field", OCCURRED_FIELDS)
def test_schema_constrains_the_string_branch_when_supported(mode, causal, field):
    model = _fact_model(_config(mode=mode, causal=causal, supports_pattern=True))
    schema = model.model_json_schema()["properties"][field]

    assert "pattern" in _string_branch(model, field)
    # null stays the escape hatch for "no derivable date" -- _infer_temporal_date
    # backfills from the text -- and the field stays optional.
    assert {"type": "null"} in schema["anyOf"]
    assert schema["default"] is None
    assert field not in model.model_json_schema().get("required", [])


@pytest.mark.parametrize(("mode", "causal"), EXTRACTION_MODES)
def test_constrained_model_keeps_the_rest_of_the_contract(mode, causal):
    """Only the two timestamp fields change; everything else is inherited."""
    baseline = _fact_model(_config(mode=mode, causal=causal, supports_pattern=False))
    constrained = _fact_model(_config(mode=mode, causal=causal, supports_pattern=True))

    assert issubclass(constrained, baseline)
    assert set(constrained.model_fields) == set(baseline.model_fields)
    assert constrained.model_json_schema().get("required") == baseline.model_json_schema().get("required")


def test_pattern_survives_the_entity_labels_schema():
    """Labels rebuild the fact model too -- the constraint must compose with it."""
    config = _config(mode="concise", causal=False, supports_pattern=True)
    config.entity_labels = [{"key": "topic", "type": "text"}]
    model = _fact_model(config)

    assert "labels" in model.model_fields
    assert "pattern" in _string_branch(model, "occurred_start")


@pytest.mark.parametrize("field", OCCURRED_FIELDS)
@pytest.mark.parametrize("value", VALID_TIMESTAMPS)
def test_pattern_accepts_real_timestamp_formats(field, value):
    model = _fact_model(_config(mode="concise", causal=True, supports_pattern=True))

    assert getattr(model.model_validate({**_minimal(model), field: value}), field) == value


@pytest.mark.parametrize("field", OCCURRED_FIELDS)
@pytest.mark.parametrize("value", INVALID_TIMESTAMPS)
def test_pattern_rejects_prose(field, value):
    model = _fact_model(_config(mode="concise", causal=True, supports_pattern=True))

    with pytest.raises(ValidationError):
        model.model_validate({**_minimal(model), field: value})


@pytest.mark.parametrize("field", OCCURRED_FIELDS)
def test_pattern_leaves_null_available(field):
    model = _fact_model(_config(mode="concise", causal=True, supports_pattern=True))

    assert getattr(model.model_validate({**_minimal(model), field: None}), field) is None


@pytest.mark.parametrize("field", OCCURRED_FIELDS)
def test_internal_fact_model_stays_lenient(field):
    """`Fact` is the post-parse storage model, not an LLM contract.

    Extraction normalises and backfills dates before building a `Fact`, and other
    call sites construct it directly, so constraining it here would reject data
    the pipeline legitimately produces.
    """
    schema = Fact.model_json_schema()["properties"][field]
    branches = schema.get("anyOf", [schema])

    assert "pattern" not in next(b for b in branches if b.get("type") == "string")
