"""
Tests for the configurable recall-budget mapping (Budget enum -> thinking_budget int).

Two functions are supported:
- "fixed": returns the recall_budget_fixed_<level> integer directly (legacy default).
- "adaptive": returns round(max_tokens * recall_budget_adaptive_<level>),
              clamped to [recall_budget_min, recall_budget_max].

Both the function selector and the per-level numbers are hierarchical config
fields (global env -> tenant -> bank), so they can be overridden per bank.
"""

import dataclasses

import pytest

from hindsight_api.config import (
    DEFAULT_RECALL_BUDGET_ADAPTIVE_HIGH,
    DEFAULT_RECALL_BUDGET_ADAPTIVE_LOW,
    DEFAULT_RECALL_BUDGET_ADAPTIVE_MID,
    DEFAULT_RECALL_BUDGET_FIXED_HIGH,
    DEFAULT_RECALL_BUDGET_FIXED_LOW,
    DEFAULT_RECALL_BUDGET_FIXED_MID,
    DEFAULT_RECALL_BUDGET_MAX,
    DEFAULT_RECALL_BUDGET_MIN,
    DEFAULT_RECALL_BUDGET_FUNCTION,
    ENV_RECALL_BUDGET_ADAPTIVE_LOW,
    ENV_RECALL_BUDGET_ADAPTIVE_MID,
    ENV_RECALL_BUDGET_FIXED_HIGH,
    ENV_RECALL_BUDGET_FIXED_LOW,
    ENV_RECALL_BUDGET_FIXED_MID,
    ENV_RECALL_BUDGET_MAX,
    ENV_RECALL_BUDGET_MIN,
    ENV_RECALL_BUDGET_FUNCTION,
    RECALL_BUDGET_FUNCTIONS,
    HindsightConfig,
)
from hindsight_api.config_resolver import _validate_recall_budget_updates
from hindsight_api.engine.memory_engine import Budget, _resolve_thinking_budget


_BUDGET_FIELD_NAMES = (
    "recall_budget_function",
    "recall_budget_fixed_low",
    "recall_budget_fixed_mid",
    "recall_budget_fixed_high",
    "recall_budget_adaptive_low",
    "recall_budget_adaptive_mid",
    "recall_budget_adaptive_high",
    "recall_budget_min",
    "recall_budget_max",
)


class TestBudgetConfigFields:
    def test_fields_exist_on_dataclass(self):
        names = {f.name for f in dataclasses.fields(HindsightConfig)}
        for field_name in _BUDGET_FIELD_NAMES:
            assert field_name in names, f"Missing dataclass field: {field_name}"

    def test_fields_are_configurable(self):
        configurable = HindsightConfig.get_configurable_fields()
        for field_name in _BUDGET_FIELD_NAMES:
            assert field_name in configurable, f"Field not in _CONFIGURABLE_FIELDS: {field_name}"

    def test_default_function_is_fixed_for_backwards_compat(self):
        # The whole point of function="fixed" being default is to preserve legacy behavior.
        assert DEFAULT_RECALL_BUDGET_FUNCTION == "fixed"
        assert "fixed" in RECALL_BUDGET_FUNCTIONS
        assert "adaptive" in RECALL_BUDGET_FUNCTIONS

    def test_default_fixed_values_match_legacy_hardcoded_mapping(self):
        # These are the values that used to live in the hardcoded budget_mapping dict.
        assert DEFAULT_RECALL_BUDGET_FIXED_LOW == 100
        assert DEFAULT_RECALL_BUDGET_FIXED_MID == 300
        assert DEFAULT_RECALL_BUDGET_FIXED_HIGH == 1000

    def test_default_adaptive_clamps_are_sane(self):
        assert DEFAULT_RECALL_BUDGET_MIN >= 1
        assert DEFAULT_RECALL_BUDGET_MAX > DEFAULT_RECALL_BUDGET_MIN

    def test_env_var_constants(self):
        assert ENV_RECALL_BUDGET_FUNCTION == "HINDSIGHT_API_RECALL_BUDGET_FUNCTION"
        assert ENV_RECALL_BUDGET_FIXED_LOW == "HINDSIGHT_API_RECALL_BUDGET_FIXED_LOW"
        assert ENV_RECALL_BUDGET_FIXED_MID == "HINDSIGHT_API_RECALL_BUDGET_FIXED_MID"
        assert ENV_RECALL_BUDGET_FIXED_HIGH == "HINDSIGHT_API_RECALL_BUDGET_FIXED_HIGH"
        assert ENV_RECALL_BUDGET_ADAPTIVE_LOW == "HINDSIGHT_API_RECALL_BUDGET_ADAPTIVE_LOW"
        assert ENV_RECALL_BUDGET_ADAPTIVE_MID == "HINDSIGHT_API_RECALL_BUDGET_ADAPTIVE_MID"
        assert ENV_RECALL_BUDGET_MIN == "HINDSIGHT_API_RECALL_BUDGET_MIN"
        assert ENV_RECALL_BUDGET_MAX == "HINDSIGHT_API_RECALL_BUDGET_MAX"

    def test_from_env_reads_overrides(self, monkeypatch):
        monkeypatch.setenv(ENV_RECALL_BUDGET_FUNCTION, "adaptive")
        monkeypatch.setenv(ENV_RECALL_BUDGET_FIXED_MID, "777")
        monkeypatch.setenv(ENV_RECALL_BUDGET_ADAPTIVE_MID, "0.5")
        monkeypatch.setenv(ENV_RECALL_BUDGET_MIN, "5")
        monkeypatch.setenv(ENV_RECALL_BUDGET_MAX, "9999")

        config = HindsightConfig.from_env()
        assert config.recall_budget_function == "adaptive"
        assert config.recall_budget_fixed_mid == 777
        assert config.recall_budget_adaptive_mid == 0.5
        assert config.recall_budget_min == 5
        assert config.recall_budget_max == 9999

    def test_from_env_invalid_function_falls_back_to_default(self, monkeypatch):
        # Defensive parsing: an invalid env value logs a warning and falls back.
        monkeypatch.setenv(ENV_RECALL_BUDGET_FUNCTION, "garbage")
        config = HindsightConfig.from_env()
        assert config.recall_budget_function == DEFAULT_RECALL_BUDGET_FUNCTION


class TestResolveThinkingBudgetFixedFunction:
    @pytest.fixture
    def fixed_config(self):
        return {
            "recall_budget_function": "fixed",
            "recall_budget_fixed_low": 100,
            "recall_budget_fixed_mid": 300,
            "recall_budget_fixed_high": 1000,
            "recall_budget_adaptive_low": 0.025,
            "recall_budget_adaptive_mid": 0.075,
            "recall_budget_adaptive_high": 0.25,
            "recall_budget_min": 20,
            "recall_budget_max": 2000,
        }

    def test_low_mid_high_match_fixed_values(self, fixed_config):
        assert _resolve_thinking_budget(fixed_config, Budget.LOW, 4096) == 100
        assert _resolve_thinking_budget(fixed_config, Budget.MID, 4096) == 300
        assert _resolve_thinking_budget(fixed_config, Budget.HIGH, 4096) == 1000

    def test_none_budget_defaults_to_mid(self, fixed_config):
        assert _resolve_thinking_budget(fixed_config, None, 4096) == 300

    def test_max_tokens_does_not_affect_fixed_function(self, fixed_config):
        # Whole point of "fixed": result is independent of max_tokens.
        assert _resolve_thinking_budget(fixed_config, Budget.MID, 1) == 300
        assert _resolve_thinking_budget(fixed_config, Budget.MID, 1_000_000) == 300

    def test_per_bank_overrides_take_effect(self, fixed_config):
        fixed_config["recall_budget_fixed_mid"] = 42
        assert _resolve_thinking_budget(fixed_config, Budget.MID, 4096) == 42


class TestResolveThinkingBudgetAdaptiveFunction:
    @pytest.fixture
    def adaptive_config(self):
        return {
            "recall_budget_function": "adaptive",
            "recall_budget_fixed_low": 100,
            "recall_budget_fixed_mid": 300,
            "recall_budget_fixed_high": 1000,
            "recall_budget_adaptive_low": 0.025,
            "recall_budget_adaptive_mid": 0.075,
            "recall_budget_adaptive_high": 0.25,
            "recall_budget_min": 20,
            "recall_budget_max": 2000,
        }

    def test_scales_with_max_tokens(self, adaptive_config):
        # 4096 * 0.075 = 307.2 -> 307
        assert _resolve_thinking_budget(adaptive_config, Budget.MID, 4096) == 307
        # 8192 * 0.075 = 614.4 -> 614
        assert _resolve_thinking_budget(adaptive_config, Budget.MID, 8192) == 614

    def test_clamps_to_floor_when_max_tokens_tiny(self, adaptive_config):
        # 100 * 0.025 = 2.5 -> 2 -> clamped to floor 20
        assert _resolve_thinking_budget(adaptive_config, Budget.LOW, 100) == 20

    def test_clamps_to_ceiling_when_max_tokens_huge(self, adaptive_config):
        # 100_000 * 0.25 = 25_000 -> clamped to ceiling 2000
        assert _resolve_thinking_budget(adaptive_config, Budget.HIGH, 100_000) == 2000

    def test_none_budget_defaults_to_mid(self, adaptive_config):
        assert _resolve_thinking_budget(adaptive_config, None, 4096) == 307

    def test_custom_clamps_per_bank(self, adaptive_config):
        adaptive_config["recall_budget_min"] = 500
        adaptive_config["recall_budget_max"] = 600
        # 4096 * 0.075 = 307 -> below floor 500
        assert _resolve_thinking_budget(adaptive_config, Budget.MID, 4096) == 500
        # 4096 * 0.25 = 1024 -> above ceiling 600
        assert _resolve_thinking_budget(adaptive_config, Budget.HIGH, 4096) == 600


class TestResolveThinkingBudgetFallbacks:
    def test_empty_config_uses_legacy_defaults(self):
        # Resilience: missing keys should not crash; fallback to legacy mapping.
        assert _resolve_thinking_budget({}, Budget.LOW, 4096) == 100
        assert _resolve_thinking_budget({}, Budget.MID, 4096) == 300
        assert _resolve_thinking_budget({}, Budget.HIGH, 4096) == 1000

    def test_unknown_function_falls_back_to_fixed(self):
        # Defensive: if some bad config slipped past validation, behave like "fixed".
        assert _resolve_thinking_budget({"recall_budget_function": "garbage"}, Budget.MID, 4096) == 300


class TestValidateRecallBudgetUpdates:
    def test_no_op_passes(self):
        _validate_recall_budget_updates({})
        _validate_recall_budget_updates({"unrelated_field": 123})

    def test_valid_function_values(self):
        _validate_recall_budget_updates({"recall_budget_function": "fixed"})
        _validate_recall_budget_updates({"recall_budget_function": "adaptive"})

    def test_invalid_function_raises(self):
        with pytest.raises(ValueError, match="recall_budget_function"):
            _validate_recall_budget_updates({"recall_budget_function": "wrong"})
        with pytest.raises(ValueError, match="recall_budget_function"):
            _validate_recall_budget_updates({"recall_budget_function": 123})

    def test_fixed_must_be_positive_integer(self):
        for key in ("recall_budget_fixed_low", "recall_budget_fixed_mid", "recall_budget_fixed_high"):
            _validate_recall_budget_updates({key: 1})
            _validate_recall_budget_updates({key: 100_000})
            with pytest.raises(ValueError, match=key):
                _validate_recall_budget_updates({key: 0})
            with pytest.raises(ValueError, match=key):
                _validate_recall_budget_updates({key: -5})
            with pytest.raises(ValueError, match=key):
                _validate_recall_budget_updates({key: 1.5})  # float not allowed for fixed
            with pytest.raises(ValueError, match=key):
                _validate_recall_budget_updates({key: True})  # bool sneaks past int check

    def test_adaptive_must_be_positive_number(self):
        for key in ("recall_budget_adaptive_low", "recall_budget_adaptive_mid", "recall_budget_adaptive_high"):
            _validate_recall_budget_updates({key: 0.001})
            _validate_recall_budget_updates({key: 1.0})
            _validate_recall_budget_updates({key: 5})  # int is acceptable as a number
            with pytest.raises(ValueError, match=key):
                _validate_recall_budget_updates({key: 0})
            with pytest.raises(ValueError, match=key):
                _validate_recall_budget_updates({key: -0.1})
            with pytest.raises(ValueError, match=key):
                _validate_recall_budget_updates({key: True})
            with pytest.raises(ValueError, match=key):
                _validate_recall_budget_updates({key: "0.5"})

    def test_min_must_be_le_max_when_both_set(self):
        _validate_recall_budget_updates({"recall_budget_min": 10, "recall_budget_max": 1000})
        _validate_recall_budget_updates({"recall_budget_min": 100, "recall_budget_max": 100})
        with pytest.raises(ValueError, match="recall_budget_min"):
            _validate_recall_budget_updates({"recall_budget_min": 5000, "recall_budget_max": 100})

    def test_min_max_must_be_positive_integers(self):
        for key in ("recall_budget_min", "recall_budget_max"):
            with pytest.raises(ValueError, match=key):
                _validate_recall_budget_updates({key: 0})
            with pytest.raises(ValueError, match=key):
                _validate_recall_budget_updates({key: -1})
            with pytest.raises(ValueError, match=key):
                _validate_recall_budget_updates({key: 1.5})
