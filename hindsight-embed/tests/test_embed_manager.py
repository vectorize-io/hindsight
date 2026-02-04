"""Tests for EmbedManager interface."""

from hindsight_embed import get_embed_manager


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
