"""Tests for forwarding commands from hindsight-embed to hindsight-cli."""

import sys
from unittest.mock import Mock

import pytest


def test_no_key_provider_is_forwarded_to_daemon(tmp_path, monkeypatch):
    """Provider implementations, not the wrapper CLI, own credential validation."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    from hindsight_embed import cli, daemon_client

    config = {"llm_provider": "mock", "llm_api_key": None}
    run_cli = Mock(return_value=0)
    monkeypatch.setattr(sys, "argv", ["hindsight-embed", "bank", "list"])
    monkeypatch.setattr(cli, "get_config", lambda: config)
    monkeypatch.setattr(daemon_client, "run_cli", run_cli)

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 0
    run_cli.assert_called_once_with(["bank", "list"], config, None)
