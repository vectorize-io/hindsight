"""Tests for lib/config.py."""

import os

import pytest

from lib.config import _cast_env, load_config


class TestConfig:
    @pytest.fixture(autouse=True)
    def _isolate_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        for key in list(os.environ):
            if key.startswith("HINDSIGHT_"):
                monkeypatch.delenv(key, raising=False)

    def test_default_recall_score_min(self):
        cfg = load_config()
        assert cfg["recallScoreMin"] == 0.25

    def test_float_env_override(self, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_RECALL_SCORE_MIN", "0.6")
        cfg = load_config()
        assert cfg["recallScoreMin"] == 0.6

    def test_invalid_float_env_is_ignored(self, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_RECALL_SCORE_MIN", "not-a-float")
        cfg = load_config()
        assert cfg["recallScoreMin"] == 0.25

    def test_cast_float(self):
        assert _cast_env("0.3", float) == 0.3
