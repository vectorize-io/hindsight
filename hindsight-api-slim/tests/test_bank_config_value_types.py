"""Tests for type validation of bank config values (write side + read side).

The bank-config API accepted `dict[str, Any]` updates without ever checking the
values against the declared ``HindsightConfig`` field types, so a client could
store a JSON object in a string-typed field such as ``observations_mission``.
The write succeeded; the bank then failed *every* consolidation with
``expected string or bytes-like object, got 'dict'`` raised by ``re.sub`` inside
prompt assembly, deterministically, forever. See issue #3218.

Two halves are covered here:
  * write side — the value is rejected with a message naming field and type;
  * read side  — a bank already carrying the bad value resolves to something
    usable instead of wedging.
"""

import pytest

from hindsight_api.config_resolver import (
    _coerce_stored_bank_overrides,
    _configurable_field_types,
    _validate_config_value_types,
)
from hindsight_api.engine.consolidation.prompts import build_consolidation_input
from hindsight_api.worker.exceptions import format_task_error


# The fields found holding JSON objects in the field report on #3218.
_STRING_FIELDS = (
    "observations_mission",
    "retain_mission",
    "reflect_mission",
    "retain_custom_instructions",
)

_BAD_MISSION = {"rules": ["only keep preferences"], "budget": 3}


class TestValidateConfigValueTypes:
    def test_every_configurable_field_has_a_type_contract(self):
        from hindsight_api.config import HindsightConfig

        # A field with no derivable contract would silently accept anything.
        assert _configurable_field_types().keys() == HindsightConfig.get_configurable_fields()

    def test_no_op_passes(self):
        _validate_config_value_types({})
        # Non-configurable keys are rejected by the caller, not here.
        _validate_config_value_types({"unrelated_field": {"any": "shape"}})

    def test_dict_in_string_field_raises(self):
        for field in _STRING_FIELDS:
            with pytest.raises(ValueError, match=field) as exc:
                _validate_config_value_types({field: _BAD_MISSION})
            # The message must name the expected and the actual type, so the 400
            # tells the caller what to send instead.
            assert "must be a string" in str(exc.value)
            assert "got dict" in str(exc.value)

    def test_valid_values_pass(self):
        _validate_config_value_types(
            {
                "observations_mission": "Keep preferences and skills.",
                "retain_chunk_size": 4000,
                "enable_observations": True,
                "mcp_enabled_tools": ["recall"],
                "memory_defense": {"mode": "off"},
                "recall_budget_adaptive_low": 0.05,
            }
        )

    def test_none_clears_override(self):
        for field in _STRING_FIELDS:
            _validate_config_value_types({field: None})

    def test_int_accepted_for_float_field(self):
        # JSON draws no int/float distinction; a ratio of 1 must not 400.
        _validate_config_value_types({"recall_budget_adaptive_high": 1})

    def test_bool_rejected_for_numeric_field(self):
        # bool is an int subclass and would sneak past a naive isinstance check.
        with pytest.raises(ValueError, match="retain_chunk_size"):
            _validate_config_value_types({"retain_chunk_size": True})

    def test_string_rejected_for_numeric_and_bool_fields(self):
        with pytest.raises(ValueError, match="retain_chunk_size"):
            _validate_config_value_types({"retain_chunk_size": "4000"})
        with pytest.raises(ValueError, match="enable_observations"):
            _validate_config_value_types({"enable_observations": "true"})

    def test_entity_labels_accepts_both_supported_shapes(self):
        # Annotated `list | None`, but parse_entity_labels() also takes the
        # {"attributes": [...]} envelope — the contract must not narrow that.
        _validate_config_value_types({"entity_labels": []})
        _validate_config_value_types({"entity_labels": {"attributes": []}})
        with pytest.raises(ValueError, match="entity_labels"):
            _validate_config_value_types({"entity_labels": "person"})


class TestCoerceStoredBankOverrides:
    def test_clean_overrides_pass_through_unchanged(self):
        overrides = {"observations_mission": "Keep preferences.", "retain_chunk_size": 4000}
        assert _coerce_stored_bank_overrides("bank1", overrides) == overrides

    def test_dict_in_string_field_is_json_encoded(self):
        coerced = _coerce_stored_bank_overrides("bank1", {"observations_mission": _BAD_MISSION})
        mission = coerced["observations_mission"]
        assert isinstance(mission, str)
        # Semantics preserved: the structure still reaches the prompt, as text.
        assert "only keep preferences" in mission

    def test_non_coercible_override_is_dropped(self):
        # An int field cannot be salvaged; the bank must fall back to the
        # server default rather than resolve to a value nothing can use.
        coerced = _coerce_stored_bank_overrides("bank1", {"retain_chunk_size": {"size": 4000}})
        assert "retain_chunk_size" not in coerced

    def test_coercion_warns_with_the_bank_and_field(self, caplog):
        with caplog.at_level("WARNING"):
            _coerce_stored_bank_overrides("bank1", {"observations_mission": _BAD_MISSION})
        assert "bank1" in caplog.text
        assert "observations_mission" in caplog.text

    def test_strategy_overrides_are_coerced_too(self):
        # apply_strategy() splices these onto the resolved config, so a nested
        # bad value wedges the bank exactly as a top-level one does.
        coerced = _coerce_stored_bank_overrides(
            "bank1",
            {"retain_strategies": {"chat": {"retain_mission": _BAD_MISSION, "retain_chunk_size": 4000}}},
        )
        chat = coerced["retain_strategies"]["chat"]
        assert isinstance(chat["retain_mission"], str)
        assert chat["retain_chunk_size"] == 4000

    def test_strategy_null_override_is_preserved(self):
        # Inside a strategy, null is a deliberate override to None (unlike a
        # top-level null, which is the "use the server default" tombstone).
        coerced = _coerce_stored_bank_overrides(
            "bank1", {"retain_strategies": {"chat": {"retain_structured_chunk_size": None}}}
        )
        assert coerced["retain_strategies"]["chat"] == {"retain_structured_chunk_size": None}

    def test_coerced_mission_survives_prompt_assembly(self):
        """The exact reported failure: a dict mission reaching prompt assembly."""
        with pytest.raises(TypeError, match="expected string or bytes-like object"):
            build_consolidation_input("facts", "observations", observations_mission=_BAD_MISSION)

        coerced = _coerce_stored_bank_overrides("bank1", {"observations_mission": _BAD_MISSION})
        prompt = build_consolidation_input(
            "facts", "observations", observations_mission=coerced["observations_mission"]
        )
        assert "only keep preferences" in prompt


class TestFormatTaskError:
    def test_message_is_prefixed_with_the_exception_class(self):
        assert format_task_error(ValueError("boom")) == "ValueError: boom"

    def test_empty_message_still_identifies_the_exception(self):
        # The reported `Task execution failed: graph_maintenance, error: ` case.
        assert format_task_error(TimeoutError()) == "TimeoutError"
