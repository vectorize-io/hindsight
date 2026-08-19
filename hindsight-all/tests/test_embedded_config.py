"""Configuration forwarding rules for HindsightEmbedded.

Regression coverage for #3253: a setting the caller does not pass must be left
out of the daemon config, so the daemon can resolve it from the profile's .env
file or the parent environment instead of receiving a client-side placeholder
that overwrites it — and that the daemon then persists back into the profile.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from hindsight import HindsightEmbedded
from hindsight_embed.daemon_embed_manager import DaemonEmbedManager

LLM_PROVIDER = "HINDSIGHT_API_LLM_PROVIDER"
LLM_API_KEY = "HINDSIGHT_API_LLM_API_KEY"
LLM_MODEL = "HINDSIGHT_API_LLM_MODEL"
LOG_LEVEL = "HINDSIGHT_API_LOG_LEVEL"
IDLE_TIMEOUT = "HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT"


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    """Isolate HOME so profile .env files never touch the real user profile.

    USERPROFILE is set as well because Path.home() consults it on Windows.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return home


def _write_profile(home, name, port, env_contents=None):
    """Create a registered profile, optionally with a pre-populated .env file."""
    profile_dir = home / ".hindsight" / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "metadata.json").write_text(
        json.dumps(
            {
                "version": 1,
                "profiles": {
                    name: {
                        "port": port,
                        "created_at": "2024-01-01T00:00:00+00:00",
                        "last_used": "2024-01-01T00:00:00+00:00",
                    }
                },
            }
        )
    )
    env_path = profile_dir / f"{name}.env"
    if env_contents is not None:
        env_path.write_text(env_contents)
    return env_path


def _daemon_env(client):
    """Run the real daemon start path with Popen stubbed, returning the child env.

    Asserting on client.config alone would not catch a regression in how the
    embed manager merges that config with the profile and the parent
    environment, which is where the reported bug actually surfaced.
    """
    manager = DaemonEmbedManager()
    captured: dict[str, dict[str, str]] = {}
    spawned = [False]

    def fake_popen(cmd, env, **kwargs):
        captured["env"] = env
        spawned[0] = True
        process = MagicMock()
        process.pid = 12345
        return process

    with (
        patch("hindsight_embed.daemon_embed_manager.subprocess.Popen", side_effect=fake_popen),
        patch("hindsight_embed.daemon_embed_manager.time.sleep"),
        patch.object(manager, "_clear_port", return_value=True),
        patch.object(manager, "_find_api_command", return_value=["hindsight-api"]),
        patch.object(manager, "is_running", side_effect=lambda profile="": spawned[0]),
        patch("hindsight_embed.daemon_embed_manager.platform.system", return_value="Linux"),
    ):
        assert manager.ensure_running(client.config, client.profile)

    return captured["env"]


def test_nothing_is_forwarded_when_nothing_is_specified(temp_home):
    assert HindsightEmbedded(profile="test").config == {}


def test_explicitly_passed_settings_are_forwarded(temp_home):
    client = HindsightEmbedded(
        profile="test",
        llm_provider="openai",
        llm_api_key="sk-real",
        llm_model="gpt-4o-mini",
        log_level="debug",
        idle_timeout=300,
    )

    assert client.config == {
        LLM_PROVIDER: "openai",
        LLM_API_KEY: "sk-real",
        LLM_MODEL: "gpt-4o-mini",
        LOG_LEVEL: "debug",
        IDLE_TIMEOUT: "300",
    }


def test_empty_api_key_is_forwarded_as_an_override(temp_home):
    """An empty string is an explicit choice, not an omission.

    Local LLM services that need no authentication rely on it to clear a key
    inherited from the environment.
    """
    assert HindsightEmbedded(profile="test", llm_api_key="").config[LLM_API_KEY] == ""


def test_idle_timeout_zero_is_forwarded(temp_home):
    """0 is falsy but meaningful ("never auto-exit"), so it must survive."""
    assert HindsightEmbedded(profile="test", idle_timeout=0).config[IDLE_TIMEOUT] == "0"


def test_omitted_key_inherits_the_parent_environment(temp_home, monkeypatch):
    monkeypatch.setenv(LLM_API_KEY, "sk-parent")
    _write_profile(temp_home, "inherit-env", 9871)

    env = _daemon_env(HindsightEmbedded(profile="inherit-env", llm_provider="openai"))

    assert env[LLM_API_KEY] == "sk-parent"


def test_omitted_settings_inherit_the_profile_env(temp_home, monkeypatch):
    for var in (LLM_PROVIDER, LLM_API_KEY, LLM_MODEL):
        monkeypatch.delenv(var, raising=False)
    env_path = _write_profile(
        temp_home,
        "prod",
        9872,
        "HINDSIGHT_API_LLM_PROVIDER=anthropic\n"
        "HINDSIGHT_API_LLM_MODEL=claude-sonnet-4-20250514\n"
        "HINDSIGHT_API_LLM_API_KEY=sk-ant-prod\n",
    )

    env = _daemon_env(HindsightEmbedded(profile="prod"))

    assert env[LLM_PROVIDER] == "anthropic"
    assert env[LLM_MODEL] == "claude-sonnet-4-20250514"
    assert env[LLM_API_KEY] == "sk-ant-prod"

    # A successful start rewrites the profile's .env; it must not come back with
    # client-side placeholders in place of the configured values.
    persisted = env_path.read_text()
    assert "HINDSIGHT_API_LLM_PROVIDER=anthropic" in persisted
    assert "HINDSIGHT_API_LLM_MODEL=claude-sonnet-4-20250514" in persisted
    assert "HINDSIGHT_API_LLM_API_KEY=sk-ant-prod" in persisted


def test_explicit_empty_key_overrides_the_parent_environment(temp_home, monkeypatch):
    monkeypatch.setenv(LLM_API_KEY, "sk-parent")
    _write_profile(temp_home, "no-auth", 9873)

    env = _daemon_env(
        HindsightEmbedded(profile="no-auth", llm_provider="lmstudio", llm_api_key="")
    )

    assert env[LLM_API_KEY] == ""


def test_explicit_settings_still_win_over_the_profile(temp_home, monkeypatch):
    monkeypatch.delenv(LLM_PROVIDER, raising=False)
    _write_profile(temp_home, "override", 9874, "HINDSIGHT_API_LLM_PROVIDER=anthropic\n")

    env = _daemon_env(HindsightEmbedded(profile="override", llm_provider="openai"))

    assert env[LLM_PROVIDER] == "openai"
