"""Strategy selection uses existing config precedence without changing defaults."""

import os

from conftest import make_user_config
from lib.config import load_config


def test_retain_strategy_is_opt_in_and_environment_overrides_user_file(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    for key in list(os.environ):
        if key.startswith("HINDSIGHT_"):
            monkeypatch.delenv(key, raising=False)

    assert load_config()["retainStrategy"] is None
    make_user_config(tmp_path, {"retainStrategy": "agent-session"})
    assert load_config()["retainStrategy"] == "agent-session"
    monkeypatch.setenv("HINDSIGHT_RETAIN_STRATEGY", "custom-session")
    assert load_config()["retainStrategy"] == "custom-session"
