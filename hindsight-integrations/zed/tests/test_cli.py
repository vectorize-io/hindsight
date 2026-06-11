"""Tests for the CLI (config scaffolding; daemon install mocked)."""

import json

import pytest

from hindsight_zed import cli


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "USER_CONFIG_FILE", tmp_path / ".hindsight" / "zed.json")
    # Don't actually touch launchd/systemd or the network during init.
    monkeypatch.setattr(cli, "_install_daemon", lambda: None)
    monkeypatch.setattr(cli, "load_config", lambda: type("C", (), {"hindsight_api_url": "x", "hindsight_api_token": None})())
    return tmp_path


def test_init_writes_config(fake_home, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["hindsight-zed", "init", "--api-token", "tok", "--api-url", "http://localhost:8888"])
    cli.main()
    written = json.loads(cli.USER_CONFIG_FILE.read_text())
    assert written["hindsightApiToken"] == "tok"
    assert written["hindsightApiUrl"] == "http://localhost:8888"


def test_init_fixed_bank(fake_home, monkeypatch):
    monkeypatch.setattr("sys.argv", ["hindsight-zed", "init", "--fixed-bank-id", "shared", "--no-daemon"])
    cli.main()
    written = json.loads(cli.USER_CONFIG_FILE.read_text())
    assert written["fixedBankId"] == "shared"


def test_init_does_not_clobber_existing_config(fake_home, monkeypatch):
    cli.USER_CONFIG_FILE.parent.mkdir(parents=True)
    cli.USER_CONFIG_FILE.write_text(json.dumps({"hindsightApiToken": "original"}))
    monkeypatch.setattr("sys.argv", ["hindsight-zed", "init", "--api-token", "new", "--no-daemon"])
    cli.main()
    assert json.loads(cli.USER_CONFIG_FILE.read_text())["hindsightApiToken"] == "original"


def test_version(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["hindsight-zed", "--version"])
    with pytest.raises(SystemExit):
        cli.main()
    assert "hindsight-zed" in capsys.readouterr().out
