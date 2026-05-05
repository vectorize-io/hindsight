"""Tests for EmbedManager interface."""

from unittest.mock import MagicMock, patch

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


def test_find_ui_command_uses_npx_yes_flag_for_published_control_plane(monkeypatch):
    """First-run UI installs must auto-confirm the published control-plane package."""
    manager = DaemonEmbedManager()
    monkeypatch.setenv("HINDSIGHT_EMBED_CP_VERSION", "9.9.9")

    with patch("pathlib.Path.exists", return_value=False):
        assert manager._find_ui_command() == [
            "npx",
            "-y",
            "@vectorize-io/hindsight-control-plane@9.9.9",
        ]


def test_find_api_command_prefers_installed_binary_over_uvx(tmp_path, monkeypatch):
    """
    When hindsight-api is installed alongside hindsight-embed (e.g. via
    `pip install hindsight-all`), _find_api_command should invoke that
    binary directly rather than shelling out to uvx. Uses sysconfig to
    locate the venv's scripts directory (issue #1401, #1240).
    """
    scripts_dir = tmp_path / "bin"
    scripts_dir.mkdir()
    api_binary = scripts_dir / "hindsight-api"
    api_binary.touch()

    manager = DaemonEmbedManager()
    # Point __file__ away from monorepo so dev-mode check doesn't trigger
    monkeypatch.setattr("hindsight_embed.daemon_embed_manager.__file__", str(tmp_path / "hindsight_embed" / "daemon_embed_manager.py"))
    monkeypatch.setattr("hindsight_embed.daemon_embed_manager.sysconfig.get_path", lambda key: str(scripts_dir))
    monkeypatch.setattr("hindsight_embed.daemon_embed_manager.platform.system", lambda: "Linux")

    assert manager._find_api_command() == [str(api_binary)]


def test_find_api_command_target_install_uses_file_relative_fallback(tmp_path, monkeypatch):
    """
    When installed with `pip install --target`, sysconfig still points at the
    system/venv scripts dir (no binary there). The __file__-relative fallback
    should find the sibling binary in <target>/bin/ (issue #1240).
    """
    # sysconfig points to an empty venv scripts dir (no binary)
    venv_scripts = tmp_path / "venv_bin"
    venv_scripts.mkdir()

    # --target layout: binary sits next to site-packages contents
    target_dir = tmp_path / "target"
    pkg_dir = target_dir / "hindsight_embed"
    pkg_dir.mkdir(parents=True)
    fake_module = pkg_dir / "daemon_embed_manager.py"
    fake_module.write_text("")
    sibling_bin = target_dir / "bin" / "hindsight-api"
    sibling_bin.parent.mkdir()
    sibling_bin.touch()

    manager = DaemonEmbedManager()
    monkeypatch.setattr("hindsight_embed.daemon_embed_manager.__file__", str(fake_module))
    monkeypatch.setattr("hindsight_embed.daemon_embed_manager.sysconfig.get_path", lambda key: str(venv_scripts))
    monkeypatch.setattr("hindsight_embed.daemon_embed_manager.platform.system", lambda: "Linux")

    assert manager._find_api_command() == [str(sibling_bin)]


def test_find_api_command_falls_back_to_uvx_when_no_binary(tmp_path, monkeypatch):
    """Without an installed binary or dev checkout, fall back to uvx."""
    scripts_dir = tmp_path / "bin"
    scripts_dir.mkdir()
    # No hindsight-api binary in scripts_dir

    manager = DaemonEmbedManager()
    monkeypatch.setattr("hindsight_embed.daemon_embed_manager.__file__", str(tmp_path / "hindsight_embed" / "daemon_embed_manager.py"))
    monkeypatch.setattr("hindsight_embed.daemon_embed_manager.sysconfig.get_path", lambda key: str(scripts_dir))
    monkeypatch.setattr("hindsight_embed.daemon_embed_manager.platform.system", lambda: "Linux")
    monkeypatch.setenv("HINDSIGHT_EMBED_API_VERSION", "1.2.3")

    assert manager._find_api_command() == ["uvx", "hindsight-api@1.2.3"]


def test_find_api_command_windows_uses_exe_suffix(tmp_path, monkeypatch):
    """On Windows, the installed binary has a .exe suffix."""
    scripts_dir = tmp_path / "Scripts"
    scripts_dir.mkdir()
    api_binary = scripts_dir / "hindsight-api.exe"
    api_binary.touch()

    manager = DaemonEmbedManager()
    # Point __file__ away from monorepo so dev-mode check doesn't trigger
    monkeypatch.setattr("hindsight_embed.daemon_embed_manager.__file__", str(tmp_path / "hindsight_embed" / "daemon_embed_manager.py"))
    monkeypatch.setattr("hindsight_embed.daemon_embed_manager.sysconfig.get_path", lambda key: str(scripts_dir))
    monkeypatch.setattr("hindsight_embed.daemon_embed_manager.platform.system", lambda: "Windows")

    assert manager._find_api_command() == [str(api_binary)]
