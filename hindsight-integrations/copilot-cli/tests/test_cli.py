"""Tests for the hindsight-copilot-cli CLI entry point."""

import json
from pathlib import Path

import pytest

from hindsight_copilot_cli.cli import main


@pytest.fixture()
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("COPILOT_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.chdir(tmp_path)
    for var in ("HINDSIGHT_API_URL", "HINDSIGHT_API_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def test_install_command_deploys(fake_home: Path) -> None:
    rc = main(["install"])
    assert rc == 0
    assert (fake_home / ".copilot" / "hindsight-copilot-cli" / "scripts" / "session_start.py").exists()
    assert (fake_home / ".copilot" / "hooks" / "hindsight-copilot-cli.json").exists()


def test_install_passes_api_url_to_user_config(fake_home: Path) -> None:
    main(["install", "--api-url", "http://localhost:8888"])
    cfg = json.loads((fake_home / ".hindsight" / "copilot-cli.json").read_text())
    assert cfg["hindsightApiUrl"] == "http://localhost:8888"


def test_install_reads_api_url_from_env(fake_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HINDSIGHT_API_URL", "http://env-url:9999")
    main(["install"])
    cfg = json.loads((fake_home / ".hindsight" / "copilot-cli.json").read_text())
    assert cfg["hindsightApiUrl"] == "http://env-url:9999"


def test_install_repo_scope(fake_home: Path) -> None:
    rc = main(["install", "--scope", "repo"])
    assert rc == 0
    assert (fake_home / ".github" / "hooks" / "hindsight-copilot-cli.json").exists()
    assert not (fake_home / ".copilot" / "hooks" / "hindsight-copilot-cli.json").exists()


def test_uninstall_command(fake_home: Path) -> None:
    main(["install"])
    rc = main(["uninstall"])
    assert rc == 0
    assert not (fake_home / ".copilot" / "hindsight-copilot-cli").exists()


def test_uninstall_repo_scope(fake_home: Path) -> None:
    main(["install", "--scope", "repo"])
    rc = main(["uninstall", "--scope", "repo"])
    assert rc == 0
    assert not (fake_home / ".github" / "hooks" / "hindsight-copilot-cli.json").exists()


def test_no_subcommand_exits_nonzero() -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code != 0
