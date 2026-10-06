"""Tests for forwarding commands from hindsight-embed to hindsight-cli."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from hindsight_embed import cli, daemon_client
from hindsight_embed.daemon_embed_manager import DaemonEmbedManager
from hindsight_embed.profile_manager import ProfileManager


@pytest.fixture
def profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProfileManager:
    """Keep profile selection and config loading independent of the developer's environment."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    for key in list(os.environ):
        if key.startswith("HINDSIGHT_"):
            monkeypatch.delenv(key)
    monkeypatch.setattr(cli, "_cli_profile_override", None)

    manager = ProfileManager()
    for name, port in (("", 18888), ("work", 19100), ("other", 19200)):
        paths = manager.resolve_profile_paths(name)
        paths.config.write_text(
            f"HINDSIGHT_API_PORT={port}\nHINDSIGHT_API_LLM_MODEL={name or 'default'}-model\n",
            encoding="utf-8",
        )
        paths.log.write_text(f"{name or 'default'} daemon log\n", encoding="utf-8")
        paths.ui_log.write_text(f"{name or 'default'} UI log\n", encoding="utf-8")
    return manager


def invoke_cli(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["hindsight-embed", *args])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 0


@pytest.mark.parametrize(
    "env_profile,flag,expected",
    [
        pytest.param(None, [], "work", id="active-profile"),
        pytest.param(None, ["-p", "other"], "other", id="cli-over-active"),
        pytest.param("work", ["--profile", "other"], "work", id="env-over-cli"),
        pytest.param(None, ["--profile", "default"], "", id="explicit-default"),
        pytest.param("default", ["--profile", "other"], "", id="env-default-over-cli"),
    ],
)
def test_forwarded_cli_uses_selected_config_and_endpoint(
    profiles: ProfileManager,
    monkeypatch: pytest.MonkeyPatch,
    env_profile: str | None,
    flag: list[str],
    expected: str,
) -> None:
    """The parsed selection must govern both daemon configuration and the Rust CLI endpoint."""
    profiles.set_active_profile("work")
    if env_profile is not None:
        monkeypatch.setenv("HINDSIGHT_EMBED_PROFILE", env_profile)
    ensure_running = Mock(return_value=True)
    child_process = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(daemon_client, "ensure_cli_installed", lambda: True)
    monkeypatch.setattr(daemon_client, "find_cli_binary", lambda: Path("hindsight"))
    monkeypatch.setattr(daemon_client, "ensure_daemon_running", ensure_running)
    monkeypatch.setattr("subprocess.run", child_process)

    invoke_cli(monkeypatch, [*flag, "bank", "list"])

    ensure_running.assert_called_once()
    config, actual_profile = ensure_running.call_args.args
    assert actual_profile == expected
    assert config["HINDSIGHT_API_LLM_MODEL"] == f"{expected or 'default'}-model"
    child_process.assert_called_once()
    assert child_process.call_args.args[0] == ["hindsight", "bank", "list"]
    assert child_process.call_args.kwargs["env"]["HINDSIGHT_API_URL"] == (
        f"http://127.0.0.1:{profiles.resolve_profile_paths(expected).port}"
    )


@pytest.mark.parametrize("command", ["daemon", "ui"])
@pytest.mark.parametrize(
    "env_profile,flag,expected",
    [
        pytest.param(None, [], "work", id="active-profile"),
        pytest.param(None, ["-p", "other"], "other", id="cli-over-active"),
        pytest.param("work", ["--profile", "other"], "work", id="env-over-cli"),
        pytest.param(None, ["--profile", "default"], "default", id="explicit-default"),
    ],
)
def test_logs_read_the_selected_profile(
    profiles: ProfileManager,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    env_profile: str | None,
    flag: list[str],
    expected: str,
) -> None:
    profiles.set_active_profile("work")
    if env_profile is not None:
        monkeypatch.setenv("HINDSIGHT_EMBED_PROFILE", env_profile)

    invoke_cli(monkeypatch, [*flag, command, "logs"])

    assert capsys.readouterr().out == f"{expected} {'UI' if command == 'ui' else 'daemon'} log\n"


def test_daemon_status_displays_the_instance_it_probes(
    profiles: ProfileManager, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    profiles.set_active_profile("work")
    monkeypatch.setenv("COLUMNS", "240")
    is_running = Mock(return_value=True)
    monkeypatch.setattr(daemon_client, "is_daemon_running", is_running)

    invoke_cli(monkeypatch, ["daemon", "status"])

    is_running.assert_called_once_with("work")
    output = capsys.readouterr().out
    assert "http://127.0.0.1:19100" in output
    assert "hindsight-embed-work" in output
    assert "http://127.0.0.1:18888" not in output


def test_daemon_start_ui_uses_the_selected_profile(profiles: ProfileManager, monkeypatch: pytest.MonkeyPatch) -> None:
    profiles.set_active_profile("work")
    monkeypatch.setattr(daemon_client, "is_daemon_running", Mock(return_value=False))
    ensure_running = Mock(return_value=True)
    monkeypatch.setattr(daemon_client, "ensure_daemon_running", ensure_running)
    start_ui = Mock(return_value=True)
    monkeypatch.setattr(DaemonEmbedManager, "start_ui", start_ui)

    invoke_cli(monkeypatch, ["daemon", "start", "--ui"])

    assert ensure_running.call_args.args[1] == "work"
    start_ui.assert_called_once_with("work", None, "0.0.0.0")


def test_profile_show_honours_explicit_default(
    profiles: ProfileManager, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    profiles.set_active_profile("work")

    invoke_cli(monkeypatch, ["--profile", "default", "profile", "show", "-o", "json"])

    result = json.loads(capsys.readouterr().out)
    assert result["name"] == "default"
    assert result["source"] == "cli_flag"
    assert result["config"] == str(profiles.resolve_profile_paths("").config)


def test_ui_status_uses_the_selected_profiles_port(
    profiles: ProfileManager, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    profiles.set_active_profile("work")
    is_running = Mock(return_value=True)
    monkeypatch.setattr(daemon_client, "is_ui_running", is_running)

    invoke_cli(monkeypatch, ["ui", "status"])

    is_running.assert_called_once_with("work", 29100)
    assert "29100" in capsys.readouterr().out


def test_configure_uses_the_creation_target_even_with_an_env_profile(
    profiles: ProfileManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HINDSIGHT_EMBED_PROFILE", "work")

    invoke_cli(
        monkeypatch,
        ["configure", "--profile", "new", "--port", "19300", "--env", "HINDSIGHT_API_LLM_MODEL=new-model"],
    )

    assert profiles.load_profile_config("new")["HINDSIGHT_API_LLM_MODEL"] == "new-model"
    assert profiles.load_profile_config("work")["HINDSIGHT_API_LLM_MODEL"] == "work-model"


def test_no_key_provider_is_forwarded_to_daemon(profiles: ProfileManager, monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider implementations, not the wrapper CLI, own credential validation."""
    config = {"llm_provider": "mock", "llm_api_key": None}
    run_cli = Mock(return_value=0)
    monkeypatch.setattr(sys, "argv", ["hindsight-embed", "bank", "list"])
    monkeypatch.setattr(cli, "get_config", lambda: config)
    monkeypatch.setattr(daemon_client, "run_cli", run_cli)

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 0
    run_cli.assert_called_once_with(["bank", "list"], config, "")
