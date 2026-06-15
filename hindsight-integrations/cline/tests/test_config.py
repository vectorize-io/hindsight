"""Configuration loading tests."""

from lib.config import _cast_env, load_config


def test_default_recall_score_min():
    cfg = load_config()
    assert cfg.recall_score_min == 0.25


def test_float_env_override(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_RECALL_SCORE_MIN", "0.6")
    cfg = load_config()
    assert cfg.recall_score_min == 0.6


def test_invalid_float_env_is_ignored(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_RECALL_SCORE_MIN", "not-a-float")
    cfg = load_config()
    assert cfg.recall_score_min == 0.25


def test_cast_float():
    assert _cast_env("0.3", float) == 0.3
