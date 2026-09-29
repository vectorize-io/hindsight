"""Tests for config loading."""

import json

from hindsight_devin_desktop.config import DEFAULT_GLOBAL_BANK, DEFAULT_HINDSIGHT_API_URL, load_config


def test_defaults(tmp_path):
    cfg = load_config(config_file=tmp_path / "missing.json", env={})
    assert cfg.hindsight_api_url == DEFAULT_HINDSIGHT_API_URL
    assert cfg.hindsight_api_token is None
    assert cfg.global_bank == DEFAULT_GLOBAL_BANK
    assert cfg.project_bank is None  # derived per-repo, not from config


def test_file_values(tmp_path):
    p = tmp_path / "devin-desktop.json"
    p.write_text(json.dumps({"hindsightApiToken": "t", "globalBank": "me"}))
    cfg = load_config(config_file=p, env={})
    assert cfg.hindsight_api_token == "t"
    assert cfg.global_bank == "me"


def test_legacy_bank_id_maps_to_global_bank(tmp_path):
    # Pre-0.2 configs stored the single bank under ``bankId``.
    p = tmp_path / "devin-desktop.json"
    p.write_text(json.dumps({"bankId": "legacy"}))
    cfg = load_config(config_file=p, env={})
    assert cfg.global_bank == "legacy"


def test_global_bank_wins_over_legacy_alias(tmp_path):
    p = tmp_path / "devin-desktop.json"
    p.write_text(json.dumps({"bankId": "legacy", "globalBank": "current"}))
    assert load_config(config_file=p, env={}).global_bank == "current"


def test_env_overrides_file(tmp_path):
    p = tmp_path / "devin-desktop.json"
    p.write_text(json.dumps({"globalBank": "from-file"}))
    cfg = load_config(
        config_file=p,
        env={"HINDSIGHT_DEVIN_DESKTOP_GLOBAL_BANK": "from-env", "HINDSIGHT_API_TOKEN": "k"},
    )
    assert cfg.global_bank == "from-env"
    assert cfg.hindsight_api_token == "k"


def test_env_sets_project_bank_override(tmp_path):
    cfg = load_config(config_file=tmp_path / "missing.json", env={"HINDSIGHT_DEVIN_DESKTOP_BANK_ID": "proj"})
    assert cfg.project_bank == "proj"


def test_malformed_file_falls_back(tmp_path):
    p = tmp_path / "devin-desktop.json"
    p.write_text("{ broken")
    assert load_config(config_file=p, env={}).global_bank == DEFAULT_GLOBAL_BANK
