"""Tests for config loading."""

import json

from hindsight_copilot.config import DEFAULT_BANK_ID, DEFAULT_HINDSIGHT_API_URL, load_config


def test_defaults(tmp_path):
    cfg = load_config(config_file=tmp_path / "missing.json", env={})
    assert cfg.hindsight_api_url == DEFAULT_HINDSIGHT_API_URL
    assert cfg.hindsight_api_token is None
    assert cfg.bank_id == DEFAULT_BANK_ID


def test_file_values(tmp_path):
    p = tmp_path / "copilot.json"
    p.write_text(json.dumps({"hindsightApiToken": "t", "bankId": "proj"}))
    cfg = load_config(config_file=p, env={})
    assert cfg.hindsight_api_token == "t"
    assert cfg.bank_id == "proj"


def test_env_overrides_file(tmp_path):
    p = tmp_path / "copilot.json"
    p.write_text(json.dumps({"bankId": "from-file"}))
    cfg = load_config(config_file=p, env={"HINDSIGHT_COPILOT_BANK_ID": "from-env", "HINDSIGHT_API_TOKEN": "k"})
    assert cfg.bank_id == "from-env"
    assert cfg.hindsight_api_token == "k"


def test_malformed_file_falls_back(tmp_path):
    p = tmp_path / "copilot.json"
    p.write_text("{ broken")
    assert load_config(config_file=p, env={}).bank_id == DEFAULT_BANK_ID
