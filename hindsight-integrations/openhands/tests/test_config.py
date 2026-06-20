"""Tests for config loading/merging."""

import json

from hindsight_openhands.config import DEFAULT_BANK_ID, DEFAULT_HINDSIGHT_API_URL, load_config


def test_defaults(tmp_path):
    cfg = load_config(config_file=tmp_path / "missing.json", env={})
    assert cfg.hindsight_api_url == DEFAULT_HINDSIGHT_API_URL
    assert cfg.hindsight_api_token is None
    assert cfg.bank_id == DEFAULT_BANK_ID


def test_file_values(tmp_path):
    p = tmp_path / "openhands.json"
    p.write_text(json.dumps({"hindsightApiUrl": "http://localhost:8888", "hindsightApiToken": "t", "bankId": "proj"}))
    cfg = load_config(config_file=p, env={})
    assert cfg.hindsight_api_url == "http://localhost:8888"
    assert cfg.hindsight_api_token == "t"
    assert cfg.bank_id == "proj"


def test_env_overrides_file(tmp_path):
    p = tmp_path / "openhands.json"
    p.write_text(json.dumps({"hindsightApiUrl": "http://file:8888", "bankId": "from-file"}))
    env = {
        "HINDSIGHT_API_URL": "http://env:9999",
        "HINDSIGHT_OPENHANDS_BANK_ID": "from-env",
        "HINDSIGHT_API_TOKEN": "k",
    }
    cfg = load_config(config_file=p, env=env)
    assert cfg.hindsight_api_url == "http://env:9999"
    assert cfg.bank_id == "from-env"
    assert cfg.hindsight_api_token == "k"


def test_malformed_file_falls_back(tmp_path):
    p = tmp_path / "openhands.json"
    p.write_text("{ not valid")
    cfg = load_config(config_file=p, env={})
    assert cfg.hindsight_api_url == DEFAULT_HINDSIGHT_API_URL
    assert cfg.bank_id == DEFAULT_BANK_ID
