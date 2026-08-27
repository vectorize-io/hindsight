"""Regression tests for the `occurred_start` / `occurred_end` extraction contract.

These two fields are LLM-facing and were typed as bare `str | None`, so the
JSON schema advertised "any string". Under grammar-constrained decoding
(`response_format: json_schema` -> GBNF) the grammar is the only thing standing
between the model and an arbitrary string -- a description is not a constraint.
When the model has to emit a timestamp it cannot derive -- e.g. a narrative
duration ("...took down the payment service for forty minutes") with no start
time -- it reasons *inside the string* and runs to the completion cap:

    "occurred_start": "2026-08-20T00:00:00Z/N/A (duration 40 mins implied ...
     Actually, I will set occurred_start/end to ... Wait, the prompt says "

That is a corrupted stored fact plus a truncated body (`finish_reason:
"length"`), which the retain path then re-sends byte-identical (#3811, #3683).

A `pattern` makes the failure structurally impossible: the grammar cannot emit
prose into the field at all, and `null` stays available as the escape hatch for
"no derivable date" (`_infer_temporal_date` already backfills from the text).
"""

import pytest
from pydantic import ValidationError

from hindsight_api.engine.retain.fact_extraction import (
    ExtractedFact,
    ExtractedFactNoCausal,
    ExtractedFactVerbose,
    Fact,
    VerbatimExtractedFact,
)

LLM_FACT_MODELS = (ExtractedFact, ExtractedFactVerbose, ExtractedFactNoCausal, VerbatimExtractedFact)

OCCURRED_FIELDS = ("occurred_start", "occurred_end")

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

# Prose that used to be accepted here. The last one is the observed runaway.
INVALID_TIMESTAMPS = (
    "N/A",
    "ongoing",
    "before Friday",
    "Starting on August 20, 2026",
    "after the incident review",
    "2026-08-20T00:00:00Z/N/A (duration 40 mins implied ... Wait, the prompt says ",
)

# Values for the required string fields; `fact_type` is a Literal.
_FIELD_STUBS = {
    "fact_type": "world",
    "what": "Marcus deployed an untested load balancer change",
}


def _minimal(model) -> dict:
    """Smallest payload satisfying this model's required fields.

    Derived from the schema rather than hard-coded: the variants differ
    (`VerbatimExtractedFact` has no `what`), and this keeps working if the
    required set changes.
    """
    required = model.model_json_schema().get("required", [])
    return {name: _FIELD_STUBS.get(name, "N/A") for name in required}


@pytest.mark.parametrize("model", LLM_FACT_MODELS)
@pytest.mark.parametrize("field", OCCURRED_FIELDS)
def test_occurred_fields_are_pattern_constrained_and_nullable(model, field):
    """The LLM-facing schema must constrain the string branch and keep null."""
    schema = model.model_json_schema()["properties"][field]

    branches = schema["anyOf"]
    string_branch = next(b for b in branches if b.get("type") == "string")
    assert "pattern" in string_branch, f"{model.__name__}.{field} advertises an unconstrained string"
    assert {"type": "null"} in branches, f"{model.__name__}.{field} must stay nullable"
    assert schema["default"] is None


@pytest.mark.parametrize("model", LLM_FACT_MODELS)
@pytest.mark.parametrize("field", OCCURRED_FIELDS)
@pytest.mark.parametrize("value", VALID_TIMESTAMPS)
def test_occurred_fields_accept_real_timestamp_formats(model, field, value):
    parsed = model.model_validate({**_minimal(model), field: value})
    assert getattr(parsed, field) == value


@pytest.mark.parametrize("model", LLM_FACT_MODELS)
@pytest.mark.parametrize("field", OCCURRED_FIELDS)
@pytest.mark.parametrize("value", INVALID_TIMESTAMPS)
def test_occurred_fields_reject_prose(model, field, value):
    with pytest.raises(ValidationError):
        model.model_validate({**_minimal(model), field: value})


@pytest.mark.parametrize("model", LLM_FACT_MODELS)
@pytest.mark.parametrize("field", OCCURRED_FIELDS)
def test_occurred_fields_still_accept_null(model, field):
    parsed = model.model_validate({**_minimal(model), field: None})
    assert getattr(parsed, field) is None


@pytest.mark.parametrize("field", OCCURRED_FIELDS)
def test_internal_fact_model_stays_lenient(field):
    """`Fact` is the post-parse storage model, not an LLM contract.

    Extraction normalises and backfills dates before building a `Fact`, and
    other call sites construct it directly, so tightening it here would reject
    data the pipeline legitimately produces. Only the LLM-facing schemas are
    constrained.
    """
    schema = Fact.model_json_schema()["properties"][field]
    branches = schema.get("anyOf", [schema])
    string_branch = next((b for b in branches if b.get("type") == "string"), None)
    assert string_branch is not None
    assert "pattern" not in string_branch
