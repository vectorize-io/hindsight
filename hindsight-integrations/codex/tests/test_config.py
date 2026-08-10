"""Tests for lib/config.py environment overrides."""

import os

import pytest
from lib.config import load_config


class TestConfigEnvOverrides:
    @pytest.fixture(autouse=True)
    def _isolate_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        for key in list(os.environ):
            if key.startswith("HINDSIGHT_"):
                monkeypatch.delenv(key, raising=False)

    def test_retain_every_n_turns_env_override_is_int(self, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_RETAIN_EVERY_N_TURNS", "3")
        cfg = load_config()
        assert cfg["retainEveryNTurns"] == 3
        assert isinstance(cfg["retainEveryNTurns"], int)

    def test_invalid_retain_every_n_turns_env_is_ignored(self, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_RETAIN_EVERY_N_TURNS", "invalid")
        cfg = load_config()
        assert cfg["retainEveryNTurns"] == 10
