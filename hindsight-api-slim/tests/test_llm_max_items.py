"""Tests for the structured-output maxItems capability flag."""

import pytest

from hindsight_api.config import ENV_LLM_SUPPORTS_MAX_ITEMS, HindsightConfig


def test_max_items_support_defaults_on(monkeypatch):
    monkeypatch.delenv(ENV_LLM_SUPPORTS_MAX_ITEMS, raising=False)
    assert HindsightConfig.from_env().llm_supports_max_items is True


def test_max_items_support_can_be_disabled(monkeypatch):
    monkeypatch.setenv(ENV_LLM_SUPPORTS_MAX_ITEMS, "false")
    assert HindsightConfig.from_env().llm_supports_max_items is False


def test_max_items_support_defaults_for_legacy_constructor_input(monkeypatch):
    monkeypatch.delenv(ENV_LLM_SUPPORTS_MAX_ITEMS, raising=False)
    legacy_values = vars(HindsightConfig.from_env()).copy()
    legacy_values.pop("llm_supports_max_items")

    assert HindsightConfig(**legacy_values).llm_supports_max_items is True


@pytest.mark.parametrize("value", ["", "yes", "tru", "disabled"])
def test_max_items_support_rejects_ambiguous_values(monkeypatch, value):
    monkeypatch.setenv(ENV_LLM_SUPPORTS_MAX_ITEMS, value)

    with pytest.raises(ValueError, match=ENV_LLM_SUPPORTS_MAX_ITEMS):
        HindsightConfig.from_env()
