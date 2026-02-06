"""Test for validating that profile environment variables are loaded correctly when starting daemon.

This is a regression test for issue #305 where profile .env files were not loaded
before daemon startup, causing environment variables to be ignored.
"""

import json
from pathlib import Path

import pytest


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    """Create a temporary home directory."""
    temp_home = tmp_path / "home"
    temp_home.mkdir()
    monkeypatch.setenv("HOME", str(temp_home))
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
        "HINDSIGHT_API_LLM_PROVIDER=openai\n" "HINDSIGHT_API_LLM_MODEL=gpt-4o-mini\n"
    )

    # Create a named profile with a DIFFERENT provider
    profile_dir = temp_home / ".hindsight" / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)

    profile_name = "myapp"
    profile_env_path = profile_dir / f"{profile_name}.env"
    profile_env_path.write_text("HINDSIGHT_API_LLM_PROVIDER=groq\n" "HINDSIGHT_API_LLM_MODEL=llama-3.1-70b\n")

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
