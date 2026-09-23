"""Runtime listener metadata must not restart an otherwise unchanged daemon."""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import hindsight_embed.daemon_embed_manager as dem
import pytest

from conftest import plugin


@pytest.fixture
def daemon_client(hermes_env: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MagicMock]:
    """Keep real profile-file I/O, but never start or stop a live daemon."""
    client = MagicMock()
    client._manager.is_running.return_value = True
    monkeypatch.setattr(plugin.HindsightMemoryProvider, "_get_client", lambda self: client)
    monkeypatch.setattr(Path, "home", lambda: hermes_env)
    monkeypatch.setenv("HERMES_HOME", str(hermes_env))
    original_console = dem.console
    try:
        yield client
    finally:
        if dem.console is not original_console:
            dem.console.file.close()
        dem.console = original_console


@pytest.mark.parametrize("runtime_port", [None, "9177"])
def test_unchanged_profile_is_not_rewritten_or_restarted(daemon_client: MagicMock, runtime_port: str | None) -> None:
    instance = plugin.HindsightMemoryProvider()
    instance._config = {"profile": "test-daemon", "llm_provider": "openai", "llm_api_key": "fixture-only"}
    profile_path = plugin._materialize_embedded_profile_env(instance._config)
    if runtime_port is not None:
        with profile_path.open("a", encoding="utf-8") as profile_file:
            profile_file.write(f"HINDSIGHT_API_PORT={runtime_port}\n")
    original = profile_path.read_bytes()

    instance._daemon_start_worker()

    daemon_client._ensure_started.assert_called_once()
    daemon_client._manager.stop.assert_not_called()
    assert profile_path.read_bytes() == original


@pytest.mark.parametrize(
    ("setting", "new_value"),
    [
        ("llm_model", "new-model"),
        ("llm_provider", "anthropic"),
        ("llm_base_url", "https://example.invalid/v1"),
        ("llm_api_key", "rotated-fixture-only"),
        ("idle_timeout", "600"),
    ],
)
def test_real_config_drift_still_rewrites_and_restarts(daemon_client: MagicMock, setting: str, new_value: str) -> None:
    instance = plugin.HindsightMemoryProvider()
    instance._config = {"profile": "test-daemon", "llm_provider": "openai", "llm_api_key": "fixture-only"}
    profile_path = plugin._materialize_embedded_profile_env(instance._config)
    with profile_path.open("a", encoding="utf-8") as profile_file:
        profile_file.write("HINDSIGHT_API_PORT=9177\n")
    instance._config[setting] = new_value

    instance._daemon_start_worker()

    daemon_client._ensure_started.assert_called_once()
    daemon_client._manager.stop.assert_called_once_with("test-daemon")
    assert plugin._load_simple_env(profile_path) == plugin._build_embedded_profile_env(instance._config)


def test_unsafe_rewrite_keeps_existing_key_and_daemon(
    daemon_client: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = plugin.HindsightMemoryProvider()
    instance._config = {"profile": "test-daemon", "llm_provider": "openai", "llm_api_key": "fixture-only"}
    profile_path = plugin._materialize_embedded_profile_env(instance._config)
    with profile_path.open("a", encoding="utf-8") as profile_file:
        profile_file.write("HINDSIGHT_API_PORT=9177\n")
    original = profile_path.read_bytes()
    instance._config["llm_model"] = "new-model"
    monkeypatch.setattr(plugin, "_may_rewrite_profile_env", lambda config: False)

    instance._daemon_start_worker()

    daemon_client._ensure_started.assert_called_once()
    daemon_client._manager.stop.assert_not_called()
    assert profile_path.read_bytes() == original
