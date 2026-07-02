"""Tests for per-operation LLM temperature configuration from environment variables.

Covers the resolution order (per-operation env -> global env -> built-in default)
and the "omit" sentinels that drop the temperature parameter for models that reject
explicit temperatures (e.g. Azure gpt-5.5 -- see issue #2459).
"""

import pytest

from hindsight_api.config import HindsightConfig, _parse_temperature

_OP_FIELDS = {
    "HINDSIGHT_API_LLM_TEMPERATURE_VERIFICATION": ("llm_temperature_verification", 0.0),
    "HINDSIGHT_API_LLM_TEMPERATURE_RETAIN": ("llm_temperature_retain", 0.1),
    "HINDSIGHT_API_LLM_TEMPERATURE_REFLECT": ("llm_temperature_reflect", 0.9),
    "HINDSIGHT_API_LLM_TEMPERATURE_CONSOLIDATION": ("llm_temperature_consolidation", 0.0),
    "HINDSIGHT_API_LLM_TEMPERATURE_BANK_MISSION": ("llm_temperature_bank_mission", 0.3),
}


def _clear_temperature_env(monkeypatch) -> None:
    monkeypatch.delenv("HINDSIGHT_API_LLM_TEMPERATURE", raising=False)
    for env_name in _OP_FIELDS:
        monkeypatch.delenv(env_name, raising=False)


def test_defaults_preserve_historical_values(monkeypatch):
    _clear_temperature_env(monkeypatch)
    config = HindsightConfig.from_env()
    for _, (field, default) in _OP_FIELDS.items():
        assert getattr(config, field) == default


def test_global_override_applies_to_all_operations(monkeypatch):
    _clear_temperature_env(monkeypatch)
    monkeypatch.setenv("HINDSIGHT_API_LLM_TEMPERATURE", "0.2")
    config = HindsightConfig.from_env()
    for _, (field, _default) in _OP_FIELDS.items():
        assert getattr(config, field) == 0.2


def test_global_none_omits_temperature_everywhere(monkeypatch):
    _clear_temperature_env(monkeypatch)
    monkeypatch.setenv("HINDSIGHT_API_LLM_TEMPERATURE", "none")
    config = HindsightConfig.from_env()
    for _, (field, _default) in _OP_FIELDS.items():
        assert getattr(config, field) is None


def test_per_operation_override_beats_global(monkeypatch):
    _clear_temperature_env(monkeypatch)
    monkeypatch.setenv("HINDSIGHT_API_LLM_TEMPERATURE", "none")
    monkeypatch.setenv("HINDSIGHT_API_LLM_TEMPERATURE_RETAIN", "0.5")
    config = HindsightConfig.from_env()
    assert config.llm_temperature_retain == 0.5
    # Other operations still follow the global "none" (omit).
    assert config.llm_temperature_reflect is None


@pytest.mark.parametrize("sentinel", ["none", "NONE", "default", "off", "unset", "", "  "])
def test_omit_sentinels(sentinel):
    assert _parse_temperature(sentinel) is None


def test_parse_temperature_rejects_out_of_range():
    with pytest.raises(ValueError):
        _parse_temperature("2.5")
    with pytest.raises(ValueError):
        _parse_temperature("-0.1")


def test_parse_temperature_rejects_non_numeric():
    with pytest.raises(ValueError):
        _parse_temperature("warm")
