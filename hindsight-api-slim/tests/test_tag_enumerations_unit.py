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


def test_retain_content_carries_tag_enumerations():
    from hindsight_api.engine.retain.types import RetainContent

    cfg = [{"namespace": "feedback", "type": "value", "values": [{"value": "a"}]}]
    rc = RetainContent(content="x", tag_enumerations=cfg)
    assert rc.tag_enumerations == cfg


def test_retain_content_default_tag_enumerations_is_none():
    from hindsight_api.engine.retain.types import RetainContent

    rc = RetainContent(content="x")
    assert rc.tag_enumerations is None


def test_build_extraction_prompt_and_schema_accepts_tag_enumerations_kwarg():
    """Kwarg threading + bank ⊕ per-retain merge should produce a config object
    in the function's internals. We don't yet assert the prompt/schema change
    (Task 5) — just that the call accepts the kwarg without error."""
    from hindsight_api.config import HindsightConfig
    from hindsight_api.engine.retain.fact_extraction import (
        _build_extraction_prompt_and_schema,
    )

    cfg = HindsightConfig.from_env()
    cfg.tag_enumerations = [
        {
            "namespace": "severity",
            "type": "value",
            "optional": True,
            "values": [{"value": "low"}, {"value": "high"}],
        }
    ]
    per_retain = [
        {
            "namespace": "feedback",
            "type": "multi-values",
            "values": [{"value": "behavior"}, {"value": "style"}],
        }
    ]
    # Just confirm the call signature accepts the kwarg and doesn't crash.
    # No assertion on prompt/schema content yet — that's Task 5.
    prompt, schema = _build_extraction_prompt_and_schema(cfg, tag_enumerations_raw=per_retain)
    assert isinstance(prompt, str) and len(prompt) > 0
    assert schema is not None


def test_build_extraction_prompt_and_schema_works_without_tag_enumerations():
    """Backwards compatibility: existing callers that don't pass the kwarg still work."""
    from hindsight_api.config import HindsightConfig
    from hindsight_api.engine.retain.fact_extraction import (
        _build_extraction_prompt_and_schema,
    )

    cfg = HindsightConfig.from_env()
    prompt, schema = _build_extraction_prompt_and_schema(cfg)
    assert isinstance(prompt, str) and len(prompt) > 0
    assert schema is not None


def test_extraction_schema_constrains_tags_when_enumerations_present():
    """When tag_enumerations are configured, the dynamic per-fact response
    schema must include a `tags` field whose values are Literal-constrained."""
    from hindsight_api.config import HindsightConfig
    from hindsight_api.engine.retain.fact_extraction import (
        _build_extraction_prompt_and_schema,
    )

    cfg = HindsightConfig.from_env()
    raw_enums = [
        {
            "namespace": "feedback",
            "type": "value",
            "optional": True,
            "values": [{"value": "behavior"}, {"value": "style"}],
        }
    ]
    prompt, schema = _build_extraction_prompt_and_schema(cfg, tag_enumerations_raw=raw_enums)
    # Prompt mentions the namespace AND at least one value
    assert "feedback" in prompt
    assert "behavior" in prompt
    assert "style" in prompt
    # JSON schema includes feedback as a constrained enum somewhere inside the
    # facts envelope. Stringify to avoid hardcoding the envelope shape.
    json_schema_text = str(schema.model_json_schema())
    assert "feedback" in json_schema_text
    assert "behavior" in json_schema_text
    assert "style" in json_schema_text


def test_extraction_schema_unchanged_when_no_enumerations():
    """Without tag_enumerations, the schema must NOT mention any enumeration
    artifacts — back-compat."""
    from hindsight_api.config import HindsightConfig
    from hindsight_api.engine.retain.fact_extraction import (
        _build_extraction_prompt_and_schema,
    )

    cfg = HindsightConfig.from_env()
    cfg.tag_enumerations = None
    prompt, schema = _build_extraction_prompt_and_schema(cfg, tag_enumerations_raw=None)
    # Smoke check: existing tests already cover the no-enum baseline; here we
    # just confirm the prompt section header isn't appended.
    assert "Enumerated tag classification" not in prompt


def test_per_retain_enumerations_appear_in_schema_when_bank_has_none():
    """The merge logic: per-retain overrides bank, but also works alone."""
    from hindsight_api.config import HindsightConfig
    from hindsight_api.engine.retain.fact_extraction import (
        _build_extraction_prompt_and_schema,
    )

    cfg = HindsightConfig.from_env()
    cfg.tag_enumerations = None
    per_retain = [
        {
            "namespace": "severity",
            "type": "multi-values",
            "values": [{"value": "low"}, {"value": "high"}],
        }
    ]
    prompt, schema = _build_extraction_prompt_and_schema(cfg, tag_enumerations_raw=per_retain)
    assert "severity" in prompt
    assert "low" in prompt
    assert "high" in prompt
    json_schema_text = str(schema.model_json_schema())
    assert "severity" in json_schema_text


def test_assignments_to_tag_strings_handles_pydantic_model_from_extractor():
    """The dynamic Pydantic model from build_tag_enumerations_response_field
    should flow through assignments_to_tag_strings correctly."""
    from hindsight_api.engine.retain.tag_enumerations import (
        assignments_to_tag_strings,
        build_tag_enumerations_response_field,
        parse_tag_enumerations,
    )

    cfg = parse_tag_enumerations(
        [
            {
                "namespace": "feedback",
                "type": "multi-values",
                "values": [{"value": "behavior"}, {"value": "style"}],
            },
        ]
    )
    field_type, _ = build_tag_enumerations_response_field(cfg)
    instance = field_type(feedback=["behavior"])
    assert assignments_to_tag_strings(instance, cfg) == ["feedback:behavior"]


def test_merge_classified_tags_helper_unions_and_sorts():
    """_merge_classified_tags merges LLM-emitted tags (validated against the
    enum vocabulary) with caller-supplied tags, deduplicates, and sorts."""
    from hindsight_api.engine.retain.fact_extraction import _merge_classified_tags
    from hindsight_api.engine.retain.tag_enumerations import parse_tag_enumerations

    cfg = parse_tag_enumerations(
        [
            {
                "namespace": "feedback",
                "type": "multi-values",
                "values": [{"value": "behavior"}, {"value": "style"}],
            },
        ]
    )
    llm_fact = {"what": "...", "tags": {"feedback": ["behavior"]}}
    merged = _merge_classified_tags(llm_fact, ["user:chris"], cfg)
    assert merged == ["feedback:behavior", "user:chris"]


def test_merge_classified_tags_helper_noop_when_no_enums():
    """If no tag_enumerations are configured, the helper must just return
    the caller-supplied tags (sorted+deduped) unchanged."""
    from hindsight_api.engine.retain.fact_extraction import _merge_classified_tags

    llm_fact = {"what": "...", "tags": {"feedback": ["behavior"]}}
    merged = _merge_classified_tags(llm_fact, ["user:chris", "team:eng"], None)
    # No enums configured => assignments_to_tag_strings returns [], so only
    # caller tags survive.
    assert merged == ["team:eng", "user:chris"]


def test_merge_classified_tags_helper_dedupes_overlap():
    """If the LLM and the caller both supply the same tag, the result must
    contain it exactly once."""
    from hindsight_api.engine.retain.fact_extraction import _merge_classified_tags
    from hindsight_api.engine.retain.tag_enumerations import parse_tag_enumerations

    cfg = parse_tag_enumerations(
        [
            {
                "namespace": "feedback",
                "type": "multi-values",
                "values": [{"value": "behavior"}],
            },
        ]
    )
    llm_fact = {"tags": {"feedback": ["behavior"]}}
    merged = _merge_classified_tags(llm_fact, ["feedback:behavior"], cfg)
    assert merged == ["feedback:behavior"]


def test_merge_classified_tags_helper_drops_out_of_vocab():
    """Defense in depth: if the LLM emits a value not in the configured
    enum vocabulary, it MUST be dropped (assignments_to_tag_strings filters)."""
    from hindsight_api.engine.retain.fact_extraction import _merge_classified_tags
    from hindsight_api.engine.retain.tag_enumerations import parse_tag_enumerations

    cfg = parse_tag_enumerations(
        [
            {
                "namespace": "feedback",
                "type": "multi-values",
                "values": [{"value": "behavior"}],
            },
        ]
    )
    llm_fact = {"tags": {"feedback": ["behavior", "made_up_value"]}}
    merged = _merge_classified_tags(llm_fact, [], cfg)
    assert merged == ["feedback:behavior"]


def test_extract_facts_from_text_merges_llm_tags_onto_fact_object(monkeypatch):
    """End-to-end at the fact-extraction layer: a stubbed LLM that returns
    a fact with `tags: {feedback: [behavior]}` should produce a Fact whose
    `tags` attribute is `["feedback:behavior"]`. Downstream
    extract_facts_from_contents merges this onto ExtractedFact.tags.

    This is a focused integration test for the LLM-to-Fact merge — the
    broader HTTP-driven integration test lives in Task 7.
    """
    import asyncio

    from hindsight_api.config import HindsightConfig
    from hindsight_api.engine.llm_wrapper import LLMConfig
    from hindsight_api.engine.retain.fact_extraction import (
        TokenUsage,
        extract_facts_from_text,
    )

    async def fake_call(self, *args, **kwargs):
        # Mirror the lenient-JSON path: return (dict, usage) when
        # return_usage=True. fact_extraction passes skip_validation=True so
        # we return raw dicts matching the dynamic schema.
        result = {
            "facts": [
                {
                    "what": "User wants ship-without-asking",
                    "when": "during chat",
                    "where": "N/A",
                    "who": "user",
                    "why": "wants speed",
                    "fact_type": "assistant",
                    "fact_kind": "conversation",
                    "tags": {"feedback": ["behavior"]},
                }
            ]
        }
        if kwargs.get("return_usage"):
            return result, TokenUsage()
        return result

    monkeypatch.setattr(LLMConfig, "call", fake_call, raising=True)

    cfg = HindsightConfig.from_env()
    # Use a minimal text so chunking yields exactly 1 chunk and we don't
    # accidentally exercise the auto-split path. The retain mode default
    # is "structured", which is what we want.
    text = "Chris said: ship without asking next time."

    per_retain_enums = [
        {
            "namespace": "feedback",
            "type": "multi-values",
            "values": [{"value": "behavior"}, {"value": "style"}],
        }
    ]

    facts, _chunks, _usage = asyncio.run(
        extract_facts_from_text(
            text=text,
            event_date=None,
            llm_config=LLMConfig.from_env(),
            agent_name="test-agent",
            config=cfg,
            context="unit test",
            metadata=None,
            tag_enumerations_raw=per_retain_enums,
        )
    )

    assert len(facts) == 1, f"expected 1 fact, got {len(facts)}: {facts}"
    fact = facts[0]
    assert fact.tags == ["feedback:behavior"], (
        f"expected LLM-emitted tags to be merged onto Fact.tags as ['feedback:behavior'], got {fact.tags!r}"
    )
