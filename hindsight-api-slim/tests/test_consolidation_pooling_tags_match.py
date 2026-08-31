"""Tests for the consolidation_pooling_tags_match config field.

Covers: env-var validation/fallback, the default value, bank-level override
resolution via ConfigResolver, and bank-template manifest validation.
"""

import pytest

from hindsight_api.api.http import BankTemplateManifest, validate_bank_template
from hindsight_api.config import (
    DEFAULT_CONSOLIDATION_POOLING_TAGS_MATCH,
    ENV_CONSOLIDATION_POOLING_TAGS_MATCH,
    HindsightConfig,
)

from .test_hierarchical_config import FakeBankConfigBackend


def test_default_value_is_all_strict(monkeypatch):
    monkeypatch.delenv(ENV_CONSOLIDATION_POOLING_TAGS_MATCH, raising=False)
    config = HindsightConfig.from_env()
    assert config.consolidation_pooling_tags_match == "all_strict" == DEFAULT_CONSOLIDATION_POOLING_TAGS_MATCH


def test_valid_env_value_any_strict(monkeypatch):
    monkeypatch.setenv(ENV_CONSOLIDATION_POOLING_TAGS_MATCH, "any_strict")
    config = HindsightConfig.from_env()
    assert config.consolidation_pooling_tags_match == "any_strict"


def test_invalid_env_value_falls_back_to_default(monkeypatch):
    # Defensive parsing: an invalid env value logs a warning and falls back,
    # rather than raising or silently propagating an unusable value.
    monkeypatch.setenv(ENV_CONSOLIDATION_POOLING_TAGS_MATCH, "garbage")
    config = HindsightConfig.from_env()
    assert config.consolidation_pooling_tags_match == DEFAULT_CONSOLIDATION_POOLING_TAGS_MATCH


def test_configurable_field_is_registered():
    assert "consolidation_pooling_tags_match" in HindsightConfig.get_configurable_fields()


@pytest.mark.asyncio
async def test_bank_override_via_config_resolver():
    from hindsight_api.config_resolver import ConfigResolver

    bank_id = "test-consolidation-pooling-tags-match-bank"
    resolver = ConfigResolver(backend=FakeBankConfigBackend())

    config = await resolver.resolve_full_config(bank_id)
    assert config.consolidation_pooling_tags_match == DEFAULT_CONSOLIDATION_POOLING_TAGS_MATCH

    await resolver.update_bank_config(bank_id, {"consolidation_pooling_tags_match": "any_strict"})
    config = await resolver.resolve_full_config(bank_id)
    assert config.consolidation_pooling_tags_match == "any_strict"

    # A JSON null override reverts to the server default, same as other fields.
    await resolver.update_bank_config(bank_id, {"consolidation_pooling_tags_match": None})
    config = await resolver.resolve_full_config(bank_id)
    assert config.consolidation_pooling_tags_match == resolver._global_config.consolidation_pooling_tags_match


def test_any_strict_is_valid_in_bank_template():
    manifest = BankTemplateManifest.model_validate(
        {
            "version": "1",
            "bank": {"consolidation_pooling_tags_match": "any_strict"},
        }
    )
    assert validate_bank_template(manifest) == []


def test_invalid_value_rejected_in_bank_template():
    manifest = BankTemplateManifest.model_validate(
        {
            "version": "1",
            "bank": {"consolidation_pooling_tags_match": "loose"},
        }
    )
    errors = validate_bank_template(manifest)
    assert any("consolidation_pooling_tags_match" in e for e in errors)
