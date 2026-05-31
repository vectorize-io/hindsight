"""Unit tests for tag_enumerations module."""

import pytest
from pydantic import ValidationError

from hindsight_api.engine.retain.tag_enumerations import (
    TagEnumeration,
    TagEnumerationsConfig,
    TagEnumValue,
    build_tag_enumerations_response_field,
    merge_tag_enumerations,
    parse_tag_enumerations,
    render_tag_enumerations_to_prompt_section,
)


def test_parse_none_returns_none():
    assert parse_tag_enumerations(None) is None


def test_parse_empty_list_returns_none():
    assert parse_tag_enumerations([]) is None


def test_parse_single_enumeration():
    raw = [
        {
            "namespace": "feedback",
            "description": "Type of correction",
            "type": "multi-values",
            "optional": True,
            "values": [
                {"value": "behavior", "description": "process / sequencing"},
                {"value": "style", "description": "tone, verbosity"},
            ],
        }
    ]
    cfg = parse_tag_enumerations(raw)
    assert cfg is not None
    assert len(cfg.enumerations) == 1
    e = cfg.enumerations[0]
    assert e.namespace == "feedback"
    assert e.type == "multi-values"
    assert len(e.values) == 2


def test_parse_rejects_unknown_type():
    raw = [{"namespace": "x", "type": "text", "values": []}]
    with pytest.raises(ValidationError):
        parse_tag_enumerations(raw)


def test_parse_rejects_empty_namespace():
    raw = [{"namespace": "", "type": "value", "values": [{"value": "a"}]}]
    with pytest.raises(ValueError):
        parse_tag_enumerations(raw)


def test_parse_rejects_enumeration_without_values():
    raw = [{"namespace": "x", "type": "value", "values": []}]
    with pytest.raises(ValueError):
        parse_tag_enumerations(raw)


def test_merge_per_retain_overrides_bank_by_namespace():
    bank = parse_tag_enumerations(
        [
            {"namespace": "feedback", "type": "value", "values": [{"value": "old"}]},
            {
                "namespace": "severity",
                "type": "value",
                "values": [{"value": "low"}, {"value": "high"}],
            },
        ]
    )
    per_retain = parse_tag_enumerations(
        [
            {
                "namespace": "feedback",
                "type": "multi-values",
                "values": [{"value": "new"}],
            },
        ]
    )
    merged = merge_tag_enumerations(bank, per_retain)
    by_ns = {e.namespace: e for e in merged.enumerations}
    assert by_ns["feedback"].type == "multi-values"
    assert by_ns["feedback"].values[0].value == "new"
    assert by_ns["severity"].values[0].value == "low"  # untouched


def test_merge_handles_none_inputs():
    cfg = parse_tag_enumerations([{"namespace": "x", "type": "value", "values": [{"value": "a"}]}])
    assert merge_tag_enumerations(None, cfg).enumerations == cfg.enumerations
    assert merge_tag_enumerations(cfg, None).enumerations == cfg.enumerations
    assert merge_tag_enumerations(None, None) is None


def test_build_response_field_value_optional():
    cfg = parse_tag_enumerations(
        [
            {
                "namespace": "feedback",
                "type": "value",
                "optional": True,
                "values": [{"value": "a"}, {"value": "b"}],
            }
        ]
    )
    field_type, field_info = build_tag_enumerations_response_field(cfg)
    from pydantic import create_model

    M = create_model("M", tags=(field_type, field_info))
    assert M(tags={"feedback": "a"}).tags.feedback == "a"
    assert M(tags={}).tags.feedback is None
    with pytest.raises(ValidationError):
        M(tags={"feedback": "zzz"})


def test_build_response_field_multi_values():
    cfg = parse_tag_enumerations(
        [
            {
                "namespace": "feedback",
                "type": "multi-values",
                "values": [{"value": "behavior"}, {"value": "style"}],
            }
        ]
    )
    field_type, field_info = build_tag_enumerations_response_field(cfg)
    from pydantic import create_model

    M = create_model("M", tags=(field_type, field_info))
    assert M(tags={"feedback": ["behavior", "style"]}).tags.feedback == [
        "behavior",
        "style",
    ]
    assert M(tags={"feedback": []}).tags.feedback == []
    with pytest.raises(ValidationError):
        M(tags={"feedback": ["nope"]})


def test_render_prompt_section_includes_namespace_and_values():
    cfg = parse_tag_enumerations(
        [
            {
                "namespace": "feedback",
                "description": "Correction type",
                "type": "multi-values",
                "values": [
                    {"value": "behavior", "description": "process / sequencing"},
                    {"value": "style", "description": "tone, verbosity"},
                ],
            }
        ]
    )
    section = render_tag_enumerations_to_prompt_section(cfg)
    assert "feedback" in section
    assert "behavior" in section
    assert "style" in section
    assert "process / sequencing" in section
    assert "multi-values" in section.lower() or "zero or more" in section.lower()


def test_render_prompt_section_empty_returns_empty_string():
    assert render_tag_enumerations_to_prompt_section(None) == ""


def test_assignments_to_tag_strings_value_and_multi():
    from hindsight_api.engine.retain.tag_enumerations import assignments_to_tag_strings

    cfg = parse_tag_enumerations(
        [
            {
                "namespace": "feedback",
                "type": "multi-values",
                "values": [{"value": "behavior"}, {"value": "style"}],
            },
            {
                "namespace": "severity",
                "type": "value",
                "values": [{"value": "low"}, {"value": "high"}],
            },
        ]
    )
    out = assignments_to_tag_strings({"feedback": ["behavior", "style"], "severity": "high"}, cfg)
    assert sorted(out) == ["feedback:behavior", "feedback:style", "severity:high"]


def test_assignments_to_tag_strings_drops_out_of_vocab():
    from hindsight_api.engine.retain.tag_enumerations import assignments_to_tag_strings

    cfg = parse_tag_enumerations(
        [
            {
                "namespace": "feedback",
                "type": "value",
                "values": [{"value": "behavior"}],
            }
        ]
    )
    out = assignments_to_tag_strings({"feedback": "nope"}, cfg)
    assert out == []


def test_assignments_to_tag_strings_drops_unknown_namespace():
    from hindsight_api.engine.retain.tag_enumerations import assignments_to_tag_strings

    cfg = parse_tag_enumerations([{"namespace": "feedback", "type": "value", "values": [{"value": "a"}]}])
    out = assignments_to_tag_strings({"unknown_ns": "a"}, cfg)
    assert out == []


def test_parse_rejects_colon_in_namespace():
    """`:` is reserved as the namespace/value separator."""
    raw = [{"namespace": "a:b", "type": "value", "values": [{"value": "x"}]}]
    with pytest.raises(ValueError):
        parse_tag_enumerations(raw)


def test_parse_accepts_dict_form():
    """parse_tag_enumerations accepts both list and {enumerations: [...]} dict."""
    raw = {
        "enumerations": [
            {"namespace": "feedback", "type": "value", "values": [{"value": "a"}]},
        ]
    }
    cfg = parse_tag_enumerations(raw)
    assert cfg is not None
    assert len(cfg.enumerations) == 1
    assert cfg.enumerations[0].namespace == "feedback"


def test_build_response_field_value_required():
    """type=value, optional=False produces a required Literal field."""
    cfg = parse_tag_enumerations(
        [
            {
                "namespace": "severity",
                "type": "value",
                "optional": False,
                "values": [{"value": "low"}, {"value": "high"}],
            }
        ]
    )
    field_type, field_info = build_tag_enumerations_response_field(cfg)
    from pydantic import create_model

    M = create_model("M", tags=(field_type, field_info))
    assert M(tags={"severity": "low"}).tags.severity == "low"
    with pytest.raises(ValidationError):
        M(tags={})  # required field missing
    with pytest.raises(ValidationError):
        M(tags={"severity": "zzz"})


def test_assignments_to_tag_strings_accepts_pydantic_model():
    """Production path: LLM response arrives as the dynamic Pydantic model."""
    from hindsight_api.engine.retain.tag_enumerations import assignments_to_tag_strings

    cfg = parse_tag_enumerations(
        [
            {
                "namespace": "feedback",
                "type": "multi-values",
                "values": [{"value": "behavior"}, {"value": "style"}],
            }
        ]
    )
    field_type, field_info = build_tag_enumerations_response_field(cfg)
    instance = field_type(feedback=["behavior", "style"])
    out = assignments_to_tag_strings(instance, cfg)
    assert sorted(out) == ["feedback:behavior", "feedback:style"]


def test_tag_enumeration_model_direct_validation():
    """The TagEnumeration model enforces its invariants when constructed directly."""
    # Happy path
    e = TagEnumeration(
        namespace="x",
        type="value",
        values=[TagEnumValue(value="a")],
    )
    assert e.namespace == "x"
    # Empty namespace rejected
    with pytest.raises(ValueError):
        TagEnumeration(namespace="", type="value", values=[TagEnumValue(value="a")])
    # No values rejected
    with pytest.raises(ValueError):
        TagEnumeration(namespace="x", type="value", values=[])


def test_tag_enumerations_config_is_constructible():
    """Smoke check that TagEnumerationsConfig is part of the public API."""
    cfg = TagEnumerationsConfig(enumerations=[])
    assert cfg.enumerations == []


def test_config_field_accepts_raw_list():
    import dataclasses

    from hindsight_api.config import HindsightConfig

    raw = [{"namespace": "feedback", "type": "value", "values": [{"value": "a"}]}]
    # HindsightConfig has hundreds of required fields; build a baseline via
    # from_env() and override only the field under test.
    base = HindsightConfig.from_env()
    cfg = dataclasses.replace(base, tag_enumerations=raw)
    # Stored as the raw list; parsing happens in the retain pipeline.
    assert cfg.tag_enumerations == raw
    # Default is None when not overridden.
    assert base.tag_enumerations is None


def test_configurable_fields_includes_tag_enumerations():
    from hindsight_api.config import HindsightConfig

    assert "tag_enumerations" in HindsightConfig._CONFIGURABLE_FIELDS
    # Also reachable via the public helper.
    assert "tag_enumerations" in HindsightConfig.get_configurable_fields()
