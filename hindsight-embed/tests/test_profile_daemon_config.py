"""Test for validating that profile environment variables are loaded correctly when starting daemon.

This is a regression test for issue #305 where profile .env files were not loaded
before daemon startup, causing environment variables to be ignored.
"""

import json
from pathlib import Path

import pytest


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    """Create a temporary home directory.

    Both POSIX and Windows env vars are set because `Path.home()` consults
    USERPROFILE (then HOMEDRIVE+HOMEPATH) on Windows, not HOME — without
    USERPROFILE override the tests would operate on the real user profile.
    """
    temp_home = tmp_path / "home"
    temp_home.mkdir()
    monkeypatch.setenv("HOME", str(temp_home))
    monkeypatch.setenv("USERPROFILE", str(temp_home))
    return temp_home


def test_profile_config_is_loaded_for_daemon(temp_home):
    """Test that profile .env config is loaded when preparing daemon startup.

    Before the fix, the daemon would ignore all values from the profile's .env file,
    using only os.environ or hardcoded defaults. This caused HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT
    and other profile-specific settings to be silently ignored.
    """
    # Create a profile with custom configuration
    profile_dir = temp_home / ".hindsight" / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)

    profile_name = "test-timeout"
    profile_env_path = profile_dir / f"{profile_name}.env"

    # Write profile config with custom values
    profile_env_path.write_text(
        "HINDSIGHT_API_LLM_PROVIDER=openai\n"
        "HINDSIGHT_API_LLM_API_KEY=sk-test-fake-key\n"
        "HINDSIGHT_API_LLM_MODEL=gpt-4o-mini\n"
        "HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT=0\n"
        "HINDSIGHT_API_LOG_LEVEL=debug\n"
    )

    # Create metadata to register the profile with a port
    metadata_path = profile_dir / "metadata.json"
    metadata = {
        "version": 1,
        "profiles": {
            profile_name: {
                "port": 9876,
                "created_at": "2024-01-01T00:00:00+00:00",
                "last_used": "2024-01-01T00:00:00+00:00",
            }
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    # Verify that ProfileManager can load the profile config
    from hindsight_embed.profile_manager import ProfileManager

    pm = ProfileManager()

    # Verify profile exists
    assert pm.profile_exists(profile_name)

    # Get profile paths
    paths = pm.resolve_profile_paths(profile_name)
    assert paths.config.exists()
    assert paths.config == profile_env_path
    assert paths.port == 9876

    # Load profile config (this simulates what the fix should do)
    profile_config = pm.load_profile_config(profile_name)

    # Verify config was loaded correctly
    assert profile_config["HINDSIGHT_API_LLM_PROVIDER"] == "openai"
    assert profile_config["HINDSIGHT_API_LLM_API_KEY"] == "sk-test-fake-key"
    assert profile_config["HINDSIGHT_API_LLM_MODEL"] == "gpt-4o-mini"
    assert profile_config["HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT"] == "0"
    assert profile_config["HINDSIGHT_API_LOG_LEVEL"] == "debug"

    # Verify that idle_timeout simple key is also available for backward compat
    # (some code checks config.get("idle_timeout"))
    assert profile_config.get("idle_timeout") == "0"


def test_load_config_file_uses_correct_profile(temp_home, monkeypatch):
    """Test that load_config_file() loads the correct profile and not default.

    This is the core fix for issue #305 - when a profile is specified, we should
    ONLY load that profile's .env, never the default profile's config.
    """
    import os

    # Clear any existing profile override state
    from hindsight_embed.cli import set_cli_profile_override

    set_cli_profile_override(None)

    # Must clear HINDSIGHT_EMBED_PROFILE env var to ensure test isolation
    monkeypatch.delenv("HINDSIGHT_EMBED_PROFILE", raising=False)

    from hindsight_embed.cli import load_config_file
    from hindsight_embed.profile_manager import ProfileManager

    # Create default profile with one provider
    default_config_dir = temp_home / ".hindsight"
    default_config_dir.mkdir(parents=True, exist_ok=True)
    (default_config_dir / "embed").write_text(
        "HINDSIGHT_API_LLM_PROVIDER=openai\nHINDSIGHT_API_LLM_MODEL=gpt-4o-mini\n"
    )

    # Create a named profile with a DIFFERENT provider
    profile_dir = temp_home / ".hindsight" / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)

    profile_name = "myapp"
    profile_env_path = profile_dir / f"{profile_name}.env"
    profile_env_path.write_text("HINDSIGHT_API_LLM_PROVIDER=groq\nHINDSIGHT_API_LLM_MODEL=llama-3.1-70b\n")

    # Create metadata
    import json

    metadata_path = profile_dir / "metadata.json"
    metadata = {
        "version": 1,
        "profiles": {
            profile_name: {
                "port": 9876,
                "created_at": "2024-01-01T00:00:00+00:00",
                "last_used": "2024-01-01T00:00:00+00:00",
            }
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    # Clear env to ensure we're testing file loading
    monkeypatch.delenv("HINDSIGHT_API_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("HINDSIGHT_API_LLM_MODEL", raising=False)

    # Test 1: Load default profile (no profile specified)
    set_cli_profile_override(None)
    load_config_file()

    assert os.environ.get("HINDSIGHT_API_LLM_PROVIDER") == "openai"
    assert os.environ.get("HINDSIGHT_API_LLM_MODEL") == "gpt-4o-mini"

    # Clear env
    monkeypatch.delenv("HINDSIGHT_API_LLM_PROVIDER")
    monkeypatch.delenv("HINDSIGHT_API_LLM_MODEL")

    # Test 2: Load named profile - should load ONLY that profile, not default
    set_cli_profile_override(profile_name)
    load_config_file()

    # Should have loaded from the named profile
    assert os.environ.get("HINDSIGHT_API_LLM_PROVIDER") == "groq", "Should load from named profile, not default"
    assert os.environ.get("HINDSIGHT_API_LLM_MODEL") == "llama-3.1-70b", "Should load from named profile, not default"


def test_get_config_respects_profile(temp_home, monkeypatch):
    """Test that get_config() returns profile-specific values."""
    import os

    from hindsight_embed.cli import get_config, set_cli_profile_override
    from hindsight_embed.profile_manager import ProfileManager

    # Create default profile
    default_config_dir = temp_home / ".hindsight"
    default_config_dir.mkdir(parents=True, exist_ok=True)
    (default_config_dir / "embed").write_text(
        "HINDSIGHT_API_LLM_PROVIDER=openai\n"
        "HINDSIGHT_API_LLM_MODEL=gpt-4o-mini\n"
        "HINDSIGHT_API_LLM_API_KEY=sk-default-key\n"
        "HINDSIGHT_EMBED_BANK_ID=default-bank\n"
    )

    # Create named profile
    profile_dir = temp_home / ".hindsight" / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)

    profile_name = "production"
    profile_env_path = profile_dir / f"{profile_name}.env"
    profile_env_path.write_text(
        "HINDSIGHT_API_LLM_PROVIDER=anthropic\n"
        "HINDSIGHT_API_LLM_MODEL=claude-sonnet-4-20250514\n"
        "HINDSIGHT_API_LLM_API_KEY=sk-ant-production\n"
        "HINDSIGHT_EMBED_BANK_ID=production-bank\n"
    )

    # Create metadata
    import json

    metadata_path = profile_dir / "metadata.json"
    metadata = {
        "version": 1,
        "profiles": {
            profile_name: {
                "port": 9900,
                "created_at": "2024-01-01T00:00:00+00:00",
                "last_used": "2024-01-01T00:00:00+00:00",
            }
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    # Clear env
    monkeypatch.delenv("HINDSIGHT_API_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("HINDSIGHT_API_LLM_MODEL", raising=False)
    monkeypatch.delenv("HINDSIGHT_API_LLM_API_KEY", raising=False)
    monkeypatch.delenv("HINDSIGHT_EMBED_BANK_ID", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Test with named profile
    set_cli_profile_override(profile_name)
    config = get_config()

    # Should have loaded from production profile, NOT default
    assert config["llm_provider"] == "anthropic", "Should use profile's provider"
    assert config["llm_model"] == "claude-sonnet-4-20250514", "Should use profile's model"
    assert config["llm_api_key"] == "sk-ant-production", "Should use profile's API key"
    assert config["bank_id"] == "production-bank", "Should use profile's bank_id"


def test_profile_env_propagates_arbitrary_hindsight_keys_to_daemon(temp_home, monkeypatch):
    """Regression test: HINDSIGHT_* keys in profile .env must reach the daemon subprocess env.

    Previously `_start_daemon` only copied a whitelist of keys (llm_*, log_level,
    idle_timeout) from the merged profile config into the daemon env. Anything else
    in the profile's .env — e.g. HINDSIGHT_API_EMBEDDINGS_PROVIDER or
    HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU on non-macOS — was silently dropped.
    """
    import json
    from unittest.mock import MagicMock, patch

    from hindsight_embed.daemon_embed_manager import DaemonEmbedManager

    # Write a profile .env containing non-whitelisted HINDSIGHT_API_* keys.
    profile_dir = temp_home / ".hindsight" / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_name = "forwarding"
    (profile_dir / f"{profile_name}.env").write_text(
        "HINDSIGHT_API_LLM_PROVIDER=openai\n"
        "HINDSIGHT_API_LLM_API_KEY=sk-x\n"
        "HINDSIGHT_API_LLM_MODEL=gpt-4o-mini\n"
        "HINDSIGHT_API_EMBEDDINGS_PROVIDER=tei\n"
        "HINDSIGHT_API_EMBEDDINGS_TEI_URL=http://localhost:8080\n"
        "HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU=1\n"
    )
    (profile_dir / "metadata.json").write_text(
        json.dumps(
            {
                "version": 1,
                "profiles": {
                    profile_name: {
                        "port": 9877,
                        "created_at": "2024-01-01T00:00:00+00:00",
                        "last_used": "2024-01-01T00:00:00+00:00",
                    }
                },
            }
        )
    )

    # Clear shell env so we're only testing profile propagation.
    for key in (
        "HINDSIGHT_API_EMBEDDINGS_PROVIDER",
        "HINDSIGHT_API_EMBEDDINGS_TEI_URL",
        "HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU",
    ):
        monkeypatch.delenv(key, raising=False)

    manager = DaemonEmbedManager()
    captured: dict[str, dict[str, str]] = {}
    popen_called = [False]

    def fake_popen(cmd, env, **kwargs):
        captured["env"] = env
        popen_called[0] = True
        proc = MagicMock()
        proc.pid = 12345
        return proc

    def fake_is_running(profile=""):
        return popen_called[0]

    # Simulate Linux so the macOS default doesn't mask the test.
    with (
        patch("hindsight_embed.daemon_embed_manager.subprocess.Popen", side_effect=fake_popen),
        patch("hindsight_embed.daemon_embed_manager.time.sleep"),
        patch.object(manager, "_clear_port", return_value=True),
        patch.object(manager, "_find_api_command", return_value=["hindsight-api"]),
        patch.object(manager, "is_running", side_effect=fake_is_running),
        patch("hindsight_embed.daemon_embed_manager.platform.system", return_value="Linux"),
    ):
        manager._start_daemon(config={}, profile=profile_name)

    env = captured["env"]
    assert env.get("HINDSIGHT_API_EMBEDDINGS_PROVIDER") == "tei"
    assert env.get("HINDSIGHT_API_EMBEDDINGS_TEI_URL") == "http://localhost:8080"
    assert env.get("HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU") == "1"


def test_daemon_child_env_var_set_in_daemon_env(temp_home, monkeypatch):
    """hindsight-embed must set _HINDSIGHT_DAEMON_CHILD=1 so the daemon child
    skips the redundant re-exec in daemonize()."""
    from unittest.mock import MagicMock, patch

    from hindsight_embed.daemon_embed_manager import DaemonEmbedManager

    manager = DaemonEmbedManager()
    captured: dict[str, dict[str, str]] = {}
    popen_called = [False]

    def fake_popen(cmd, env, **kwargs):
        captured["env"] = env
        popen_called[0] = True
        proc = MagicMock()
        proc.pid = 12345
        return proc

    def fake_is_running(profile=""):
        return popen_called[0]

    with (
        patch("hindsight_embed.daemon_embed_manager.subprocess.Popen", side_effect=fake_popen),
        patch("hindsight_embed.daemon_embed_manager.time.sleep"),
        patch.object(manager, "_clear_port", return_value=True),
        patch.object(manager, "_find_api_command", return_value=["hindsight-api"]),
        patch.object(manager, "is_running", side_effect=fake_is_running),
    ):
        manager._start_daemon(
            config={"llm_provider": "openai", "llm_api_key": "sk-x", "llm_model": "gpt-4o-mini"},
            profile="",
        )

    env = captured["env"]
    assert env.get("_HINDSIGHT_DAEMON_CHILD") == "1"


def test_windows_popen_uses_detached_process_flags(temp_home, monkeypatch):
    """
    On Windows the daemon must be spawned with
    `creationflags=DETACHED_PROCESS|CREATE_NEW_PROCESS_GROUP` (the POSIX
    `start_new_session` doesn't exist there) AND stdout/stderr must be
    redirected, since DETACHED_PROCESS leaves the child with no console.
    """
    import subprocess
    from unittest.mock import MagicMock, patch

    from hindsight_embed.daemon_embed_manager import DaemonEmbedManager

    # These subprocess constants only exist on Windows CPython; patch them in
    # so the test passes on Linux/macOS CI too. Values from Win32 API docs.
    monkeypatch.setattr(subprocess, "DETACHED_PROCESS", 0x00000008, raising=False)
    monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200, raising=False)

    manager = DaemonEmbedManager()
    captured: dict = {}
    popen_called = [False]

    def fake_popen(cmd, env, **kwargs):
        captured["kwargs"] = kwargs
        popen_called[0] = True
        proc = MagicMock()
        proc.pid = 12345
        return proc

    def fake_is_running(profile=""):
        return popen_called[0]

    with (
        patch("hindsight_embed.daemon_embed_manager.subprocess.Popen", side_effect=fake_popen),
        patch("hindsight_embed.daemon_embed_manager.time.sleep"),
        patch.object(manager, "_clear_port", return_value=True),
        patch.object(manager, "_find_api_command", return_value=["hindsight-api"]),
        patch.object(manager, "is_running", side_effect=fake_is_running),
        patch("hindsight_embed.daemon_embed_manager.platform.system", return_value="Windows"),
    ):
        manager._start_daemon(
            config={"llm_provider": "openai", "llm_api_key": "sk-x", "llm_model": "gpt-4o-mini"},
            profile="",
        )

    kwargs = captured["kwargs"]
    expected_flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    assert kwargs.get("creationflags") == expected_flags
    assert "start_new_session" not in kwargs
    assert kwargs.get("stdin") is subprocess.DEVNULL
    assert kwargs.get("stderr") is subprocess.STDOUT
    # stdout must be a real file handle, not None — a None stdout under
    # DETACHED_PROCESS crashes the child on first write.
    assert kwargs.get("stdout") is not None


def test_posix_popen_uses_start_new_session(temp_home, monkeypatch):
    """Lock down POSIX behavior so a future refactor doesn't flip it."""
    from unittest.mock import MagicMock, patch

    from hindsight_embed.daemon_embed_manager import DaemonEmbedManager

    manager = DaemonEmbedManager()
    captured: dict = {}
    popen_called = [False]

    def fake_popen(cmd, env, **kwargs):
        captured["kwargs"] = kwargs
        popen_called[0] = True
        proc = MagicMock()
        proc.pid = 12345
        return proc

    def fake_is_running(profile=""):
        return popen_called[0]

    with (
        patch("hindsight_embed.daemon_embed_manager.subprocess.Popen", side_effect=fake_popen),
        patch("hindsight_embed.daemon_embed_manager.time.sleep"),
        patch.object(manager, "_clear_port", return_value=True),
        patch.object(manager, "_find_api_command", return_value=["hindsight-api"]),
        patch.object(manager, "is_running", side_effect=fake_is_running),
        patch("hindsight_embed.daemon_embed_manager.platform.system", return_value="Linux"),
    ):
        manager._start_daemon(
            config={"llm_provider": "openai", "llm_api_key": "sk-x", "llm_model": "gpt-4o-mini"},
            profile="",
        )

    kwargs = captured["kwargs"]
    assert kwargs.get("start_new_session") is True
    assert "creationflags" not in kwargs


def test_posix_popen_redirects_stdout_stderr_to_log(temp_home, monkeypatch):
    """Regression test for #1380: on POSIX, the daemon child must NOT inherit
    the parent's stdout/stderr — output would otherwise leak into a TUI
    parent communicating over stdio (Hermes, JSON-RPC gateways) and corrupt
    its rendering. Both streams must be redirected to a real file handle.
    """
    from unittest.mock import MagicMock, patch

    from hindsight_embed.daemon_embed_manager import DaemonEmbedManager

    manager = DaemonEmbedManager()
    captured: dict = {}
    popen_called = [False]

    def fake_popen(cmd, env, **kwargs):
        captured["kwargs"] = kwargs
        popen_called[0] = True
        proc = MagicMock()
        proc.pid = 12345
        return proc

    def fake_is_running(profile=""):
        return popen_called[0]

    with (
        patch("hindsight_embed.daemon_embed_manager.subprocess.Popen", side_effect=fake_popen),
        patch("hindsight_embed.daemon_embed_manager.time.sleep"),
        patch.object(manager, "_clear_port", return_value=True),
        patch.object(manager, "_find_api_command", return_value=["hindsight-api"]),
        patch.object(manager, "is_running", side_effect=fake_is_running),
        patch("hindsight_embed.daemon_embed_manager.platform.system", return_value="Linux"),
    ):
        manager._start_daemon(
            config={"llm_provider": "openai", "llm_api_key": "sk-x", "llm_model": "gpt-4o-mini"},
            profile="",
        )

    kwargs = captured["kwargs"]
    stdout = kwargs.get("stdout")
    stderr = kwargs.get("stderr")
    # Must be real file objects, not None (inherit) or DEVNULL.
    assert stdout is not None and hasattr(stdout, "write"), (
        f"daemon stdout must be redirected to a file handle, got {stdout!r}"
    )
    assert stderr is not None and hasattr(stderr, "write"), (
        f"daemon stderr must be redirected to a file handle, got {stderr!r}"
    )


def test_get_config_does_not_default_llm_model(temp_home, monkeypatch):
    """Regression test for issue #1360.

    When `HINDSIGHT_API_LLM_MODEL` is unset, `get_config()` must return None
    rather than a hardcoded `gpt-4o-mini`. Previously the CLI tried to import
    `hindsight_api.config.PROVIDER_DEFAULT_MODELS` to compute a provider-keyed
    default, but that import fails in standalone venvs (`uvx hindsight-embed`,
    bundled installs) where `hindsight-api` isn't installed, and the fallback
    silently routed every non-OpenAI provider to `gpt-4o-mini` — which they
    reject, causing retain to silently store zero memories.

    Leaving the value unset lets the daemon (which has hindsight-api) resolve
    the correct provider default itself.
    """
    from hindsight_embed.cli import get_config, set_cli_profile_override

    set_cli_profile_override(None)
    monkeypatch.delenv("HINDSIGHT_EMBED_PROFILE", raising=False)
    monkeypatch.delenv("HINDSIGHT_API_LLM_MODEL", raising=False)
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "gemini")

    config = get_config()

    assert config["llm_provider"] == "gemini"
    assert config["llm_model"] is None, (
        "llm_model must be None when env var is unset so the daemon resolves "
        "the provider default; got a hardcoded fallback instead"
    )


def test_configure_from_env_omits_model_when_unset(temp_home, monkeypatch):
    """Regression test for issue #1360.

    `_do_configure_from_env` must not write a hardcoded `HINDSIGHT_API_LLM_MODEL`
    line into the profile .env when the user didn't set one — that line would
    then be re-injected on every daemon start and suppress the daemon's
    provider-keyed default lookup.
    """
    from hindsight_embed import cli

    # CONFIG_DIR/CONFIG_FILE are computed from Path.home() at module import time,
    # so the temp_home fixture's HOME override doesn't reach them. Redirect the
    # module-level constants for the duration of this test.
    config_dir = temp_home / ".hindsight"
    monkeypatch.setattr(cli, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "CONFIG_FILE", config_dir / "embed")

    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("HINDSIGHT_API_LLM_API_KEY", "test-key")
    monkeypatch.delenv("HINDSIGHT_API_LLM_MODEL", raising=False)

    rc = cli._do_configure_from_env()
    assert rc == 0

    contents = (config_dir / "embed").read_text()
    assert "HINDSIGHT_API_LLM_PROVIDER=gemini" in contents
    assert "HINDSIGHT_API_LLM_MODEL" not in contents, (
        "model line must be omitted when the user didn't set one, so the daemon picks the provider default"
    )


def test_configure_from_env_accepts_providers_outside_interactive_menu(temp_home, monkeypatch):
    """Regression test for issue #1360.

    `_do_configure_from_env` previously rejected any provider not in the small
    interactive-menu set (`PROVIDER_API_KEYS` — 5 entries) with "Unknown
    provider". hindsight-api supports ~18 providers (anthropic, claude-code,
    bedrock, openrouter, ...), so the gate blocked valid CI configurations.
    Validation belongs in the daemon, not in the CLI's UX-only menu list.
    """
    from hindsight_embed import cli

    config_dir = temp_home / ".hindsight"
    monkeypatch.setattr(cli, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "CONFIG_FILE", config_dir / "embed")

    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("HINDSIGHT_API_LLM_API_KEY", "sk-ant-test")
    monkeypatch.delenv("HINDSIGHT_API_LLM_MODEL", raising=False)

    rc = cli._do_configure_from_env()
    assert rc == 0, "anthropic must be accepted — the daemon validates providers, not the CLI"

    contents = (config_dir / "embed").read_text()
    assert "HINDSIGHT_API_LLM_PROVIDER=anthropic" in contents
