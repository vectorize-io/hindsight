"""Tests for config loading."""

import json

from hindsight_aider.config import DEFAULT_HINDSIGHT_API_URL, load_config


def test_defaults(tmp_path):
    cfg = load_config(config_file=tmp_path / "missing.json", env={})
    assert cfg.hindsight_api_url == DEFAULT_HINDSIGHT_API_URL
    assert cfg.bank_id is None
    assert cfg.auto_recall is True and cfg.auto_retain is True


def test_file_values(tmp_path):
    p = tmp_path / "aider.json"
    p.write_text(json.dumps({"hindsightApiToken": "t", "bankId": "b", "autoRetain": False}))
    cfg = load_config(config_file=p, env={})
    assert cfg.hindsight_api_token == "t"
    assert cfg.bank_id == "b"
    assert cfg.auto_retain is False


def test_env_overrides_file(tmp_path):
    p = tmp_path / "aider.json"
    p.write_text(json.dumps({"bankId": "from-file"}))
    cfg = load_config(
        config_file=p, env={"HINDSIGHT_AIDER_BANK_ID": "from-env", "HINDSIGHT_AIDER_AUTO_RECALL": "false"}
    )
    assert cfg.bank_id == "from-env"
    assert cfg.auto_recall is False


def test_recall_preamble_overridable(tmp_path):
    p = tmp_path / "aider.json"
    p.write_text(json.dumps({"recallPreamble": "From file:"}))
    assert load_config(config_file=p, env={}).recall_preamble == "From file:"
    cfg = load_config(config_file=p, env={"HINDSIGHT_AIDER_RECALL_PREAMBLE": "From env:"})
    assert cfg.recall_preamble == "From env:"


def test_malformed_file_falls_back(tmp_path):
    p = tmp_path / "aider.json"
    p.write_text("{ broken")
    cfg = load_config(config_file=p, env={})
    assert cfg.hindsight_api_url == DEFAULT_HINDSIGHT_API_URL
