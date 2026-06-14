"""Tests for config loading/merging."""

import json

from hindsight_zed.config import DEFAULT_HINDSIGHT_API_URL, load_config


def test_defaults(tmp_path):
    cfg = load_config(config_file=tmp_path / "nope.json", env={})
    assert cfg.hindsight_api_url == DEFAULT_HINDSIGHT_API_URL
    assert cfg.auto_recall is True
    assert cfg.auto_retain is True
    assert cfg.bank_prefix == "zed"
    assert cfg.fixed_bank_id is None


def test_file_overrides(tmp_path):
    f = tmp_path / "zed.json"
    f.write_text(json.dumps({"hindsightApiUrl": "http://localhost:8888", "fixedBankId": "shared", "autoRecall": False}))
    cfg = load_config(config_file=f, env={})
    assert cfg.hindsight_api_url == "http://localhost:8888"
    assert cfg.fixed_bank_id == "shared"
    assert cfg.auto_recall is False


def test_env_overrides_file(tmp_path):
    f = tmp_path / "zed.json"
    f.write_text(json.dumps({"hindsightApiUrl": "http://from-file:8888"}))
    cfg = load_config(config_file=f, env={"HINDSIGHT_API_URL": "http://from-env:9999", "HINDSIGHT_ZED_DEBUG": "true"})
    assert cfg.hindsight_api_url == "http://from-env:9999"
    assert cfg.debug is True


def test_empty_url_falls_back_to_cloud(tmp_path):
    f = tmp_path / "zed.json"
    f.write_text(json.dumps({"hindsightApiUrl": ""}))
    cfg = load_config(config_file=f, env={})
    assert cfg.hindsight_api_url == DEFAULT_HINDSIGHT_API_URL


def test_bad_types_ignored(tmp_path):
    f = tmp_path / "zed.json"
    f.write_text(json.dumps({"recallMaxTokens": "not-an-int"}))
    cfg = load_config(config_file=f, env={})
    assert cfg.recall_max_tokens == 1024  # default kept
