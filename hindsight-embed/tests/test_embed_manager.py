"""Tests for EmbedManager interface."""

from unittest.mock import MagicMock

from hindsight_embed import get_embed_manager
from hindsight_embed.daemon_embed_manager import DaemonEmbedManager


def test_sanitize_profile_name_via_db_url():
    """Test profile name sanitization through database URL generation."""
    manager = get_embed_manager()

    # Test None defaults to "default"
    assert manager.get_database_url(None) == "pg0://hindsight-embed-default"

    # Test simple alphanumeric names
    assert manager.get_database_url("myapp") == "pg0://hindsight-embed-myapp"
    assert manager.get_database_url("my-app") == "pg0://hindsight-embed-my-app"
    assert manager.get_database_url("my_app") == "pg0://hindsight-embed-my_app"
    assert manager.get_database_url("app123") == "pg0://hindsight-embed-app123"

    # Test special characters get replaced with dashes
    assert manager.get_database_url("my app") == "pg0://hindsight-embed-my-app"
    assert manager.get_database_url("my.app") == "pg0://hindsight-embed-my-app"
    assert manager.get_database_url("my@app!") == "pg0://hindsight-embed-my-app-"
    assert manager.get_database_url("My App 2.0!") == "pg0://hindsight-embed-My-App-2-0-"


def test_get_database_url_default():
    """Test database URL generation with default pg0."""
    manager = get_embed_manager()

    assert manager.get_database_url("myapp") == "pg0://hindsight-embed-myapp"
    assert manager.get_database_url("myapp", None) == "pg0://hindsight-embed-myapp"
    assert manager.get_database_url("myapp", "pg0") == "pg0://hindsight-embed-myapp"


def test_get_database_url_custom():
    """Test database URL generation with custom database."""
    manager = get_embed_manager()

    custom_url = "postgresql://user:pass@localhost/db"
    assert manager.get_database_url("myapp", custom_url) == custom_url
    assert manager.get_database_url("any-profile", custom_url) == custom_url


def test_manager_singleton():
    """Test that get_embed_manager returns functional instances."""
    manager1 = get_embed_manager()
    manager2 = get_embed_manager()

    # They should be independent instances but same type
    assert type(manager1) == type(manager2)

    # They should produce the same results
    assert manager1.get_database_url("test") == manager2.get_database_url("test")


def test_register_profile_skips_when_no_api_keys():
    """
    When config contains only short keys (no HINDSIGHT_API_* prefix),
    _register_profile should not call create_profile, preserving any
    existing profile .env file.

    Regression test for https://github.com/vectorize-io/hindsight/issues/894
    """
    manager = DaemonEmbedManager()
    manager._profile_manager = MagicMock()

    # Config with short keys (as passed from cli.py's get_config())
    config = {"llm_api_key": "sk-123", "llm_provider": "openai", "llm_model": "gpt-4o"}
    manager._register_profile("myprofile", 8100, config)

    manager._profile_manager.create_profile.assert_not_called()


def test_register_profile_calls_create_when_api_keys_present():
    """
    When config contains HINDSIGHT_API_* keys, _register_profile should
    forward them to create_profile.
    """
    manager = DaemonEmbedManager()
    manager._profile_manager = MagicMock()

    config = {
        "HINDSIGHT_API_LLM_PROVIDER": "openai",
        "HINDSIGHT_API_LLM_API_KEY": "sk-123",
        "some_internal_key": "ignored",
    }
    manager._register_profile("myprofile", 8100, config)

    manager._profile_manager.create_profile.assert_called_once_with(
        "myprofile",
        8100,
        {"HINDSIGHT_API_LLM_PROVIDER": "openai", "HINDSIGHT_API_LLM_API_KEY": "sk-123"},
    )
