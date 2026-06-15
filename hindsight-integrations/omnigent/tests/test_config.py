"""Unit tests for Hindsight Omnigent configuration."""

import pytest
from hindsight_omnigent import (
    HindsightOmnigentConfig,
    configure,
    get_config,
    reset_config,
)
from hindsight_omnigent.config import DEFAULT_HINDSIGHT_API_URL


@pytest.fixture(autouse=True)
def _clean_config():
    reset_config()
    yield
    reset_config()


def test_defaults():
    cfg = configure()
    assert isinstance(cfg, HindsightOmnigentConfig)
    assert cfg.hindsight_api_url == DEFAULT_HINDSIGHT_API_URL
    assert cfg.api_key is None
    assert cfg.bank_id is None
    assert cfg.budget == "mid"
    assert cfg.max_tokens == 4096
    assert cfg.recall_tags_match == "any"


def test_explicit_values_win():
    cfg = configure(
        hindsight_api_url="http://localhost:8888",
        api_key="hsk_x",
        bank_id="b1",
        budget="high",
        max_tokens=2048,
        tags=["t"],
        recall_tags=["r"],
        recall_tags_match="all_strict",
    )
    assert cfg.hindsight_api_url == "http://localhost:8888"
    assert cfg.api_key == "hsk_x"
    assert cfg.bank_id == "b1"
    assert cfg.budget == "high"
    assert cfg.max_tokens == 2048
    assert cfg.tags == ["t"]
    assert cfg.recall_tags == ["r"]
    assert cfg.recall_tags_match == "all_strict"


def test_env_fallbacks(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_URL", "http://env:9999")
    monkeypatch.setenv("HINDSIGHT_API_KEY", "hsk_env")
    monkeypatch.setenv("HINDSIGHT_BANK_ID", "env-bank")
    cfg = configure()
    assert cfg.hindsight_api_url == "http://env:9999"
    assert cfg.api_key == "hsk_env"
    assert cfg.bank_id == "env-bank"


def test_explicit_overrides_env(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_BANK_ID", "env-bank")
    cfg = configure(bank_id="explicit")
    assert cfg.bank_id == "explicit"


def test_get_config_auto_creates_from_env(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_BANK_ID", "lazy-bank")
    # No prior configure() call — get_config() should build one from env.
    cfg = get_config()
    assert cfg.bank_id == "lazy-bank"


def test_reset_config():
    configure(bank_id="b")
    reset_config()
    # get_config rebuilds from env (none set) -> bank_id None
    assert get_config().bank_id is None
