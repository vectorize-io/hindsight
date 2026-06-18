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


def test_find_ui_command_uses_npx_yes_flag_when_npx_not_on_path(monkeypatch):
    """When npx is not on PATH, fall back to a bare `npx -y` command so the
    surrounding FileNotFoundError handler can report a clean install hint."""
    manager = DaemonEmbedManager()

    with patch("pathlib.Path.exists", return_value=False), patch("shutil.which", return_value=None):
        assert manager._find_ui_command("9.9.9") == [
            "npx",
            "-y",
            "@vectorize-io/hindsight-control-plane@9.9.9",
        ]


def test_find_ui_command_resolves_npx_absolute_path_with_yes_flag(monkeypatch):
    """When npx is on PATH, use the resolved absolute path (Windows detached
    processes don't always inherit PATH — see embed_manager._find_ui_command).
    Either way, `-y` must be set so first-run installs don't block on a prompt."""
    manager = DaemonEmbedManager()

    with patch("pathlib.Path.exists", return_value=False), patch("shutil.which", return_value="/usr/local/bin/npx"):
        cmd = manager._find_ui_command("9.9.9")

    assert cmd == [
        "/usr/local/bin/npx",
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
    monkeypatch.setattr(
        "hindsight_embed.daemon_embed_manager.__file__", str(tmp_path / "hindsight_embed" / "daemon_embed_manager.py")
    )
    monkeypatch.setattr("hindsight_embed.daemon_embed_manager.sysconfig.get_path", lambda key: str(scripts_dir))
    monkeypatch.setattr("hindsight_embed.daemon_embed_manager.platform.system", lambda: "Linux")

    assert manager._find_api_command("0.0.0") == [str(api_binary)]


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

    assert manager._find_api_command("0.0.0") == [str(sibling_bin)]


def test_find_api_command_falls_back_to_uvx_when_no_binary(tmp_path, monkeypatch):
    """Without an installed binary or dev checkout, fall back to uvx."""
    scripts_dir = tmp_path / "bin"
    scripts_dir.mkdir()
    # No hindsight-api binary in scripts_dir

    manager = DaemonEmbedManager()
    monkeypatch.setattr(
        "hindsight_embed.daemon_embed_manager.__file__", str(tmp_path / "hindsight_embed" / "daemon_embed_manager.py")
    )
    monkeypatch.setattr("hindsight_embed.daemon_embed_manager.sysconfig.get_path", lambda key: str(scripts_dir))
    monkeypatch.setattr("hindsight_embed.daemon_embed_manager.platform.system", lambda: "Linux")

    assert manager._find_api_command("1.2.3") == ["uvx", "hindsight-api@1.2.3"]


def test_find_api_command_windows_uses_exe_suffix(tmp_path, monkeypatch):
    """On Windows, the installed console binary has a .exe suffix.

    Pin sys.executable to an interpreter dir without a pythonw.exe sibling so the
    GUI-interpreter swap (issue #1885) is skipped and we deterministically
    exercise the console-exe fallback — the path that proves .exe-suffix
    resolution. The pythonw swap itself is covered below and in
    test_profile_daemon_config.py.
    """
    scripts_dir = tmp_path / "Scripts"
    scripts_dir.mkdir()
    api_binary = scripts_dir / "hindsight-api.exe"
    api_binary.touch()
    interp_dir = tmp_path / "interp"
    interp_dir.mkdir()
    (interp_dir / "python.exe").touch()  # deliberately no pythonw.exe sibling

    manager = DaemonEmbedManager()
    # Point __file__ away from monorepo so dev-mode check doesn't trigger
    monkeypatch.setattr(
        "hindsight_embed.daemon_embed_manager.__file__", str(tmp_path / "hindsight_embed" / "daemon_embed_manager.py")
    )
    monkeypatch.setattr("hindsight_embed.daemon_embed_manager.sysconfig.get_path", lambda key: str(scripts_dir))
    monkeypatch.setattr("hindsight_embed.daemon_embed_manager.platform.system", lambda: "Windows")
    monkeypatch.setattr("hindsight_embed.daemon_embed_manager.sys.executable", str(interp_dir / "python.exe"))

    assert manager._find_api_command("0.0.0") == [str(api_binary)]


def test_find_api_command_windows_prefers_gui_interpreter(tmp_path, monkeypatch):
    """On Windows, launch via pythonw.exe instead of the console exe (issue #1885).

    The console-subsystem hindsight-api.exe makes Windows Terminal's ConPTY pop a
    visible tab on daemon start; the GUI-subsystem pythonw.exe never allocates a
    console. When pythonw.exe sits next to sys.executable, _find_api_command must
    return `pythonw.exe -m hindsight_api.main`.
    """
    scripts_dir = tmp_path / "Scripts"
    scripts_dir.mkdir()
    (scripts_dir / "hindsight-api.exe").touch()
    (scripts_dir / "python.exe").touch()
    pythonw = scripts_dir / "pythonw.exe"
    pythonw.touch()

    manager = DaemonEmbedManager()
    # Point __file__ away from monorepo so dev-mode check doesn't trigger
    monkeypatch.setattr(
        "hindsight_embed.daemon_embed_manager.__file__", str(tmp_path / "hindsight_embed" / "daemon_embed_manager.py")
    )
    monkeypatch.setattr("hindsight_embed.daemon_embed_manager.sysconfig.get_path", lambda key: str(scripts_dir))
    monkeypatch.setattr("hindsight_embed.daemon_embed_manager.platform.system", lambda: "Windows")
    monkeypatch.setattr("hindsight_embed.daemon_embed_manager.sys.executable", str(scripts_dir / "python.exe"))

    assert manager._find_api_command("0.0.0") == [str(pythonw), "-m", "hindsight_api.main"]


def test_find_pid_on_port_windows_hides_netstat_console(monkeypatch):
    """Windows netstat probes must not flash a console window."""
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        result = MagicMock()
        result.returncode = 0
        result.stdout = "  TCP    127.0.0.1:9177    0.0.0.0:0    LISTENING    4321\n"
        return result

    monkeypatch.setattr("hindsight_embed.daemon_embed_manager.platform.system", lambda: "Windows")
    monkeypatch.setattr("hindsight_embed.daemon_embed_manager.subprocess.CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr("hindsight_embed.daemon_embed_manager.subprocess.run", fake_run)

    assert DaemonEmbedManager._find_pid_on_port(9177) == 4321
    assert calls[0][1]["creationflags"] == 0x08000000


def test_stop_ui_kills_recorded_and_configured_ports(tmp_path, monkeypatch):
    """After a UI-port change, stop_ui must kill BOTH the recorded (old, actually
    running) port and the configured (new) port — otherwise the old UI orphans."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    manager = DaemonEmbedManager()
    paths = manager._profile_manager.resolve_profile_paths("")  # default profile
    manager._record_ui_port(paths, 9000)  # UI was actually started on 9000
    assert manager._ui_port_file(paths).exists()

    killed = []
    monkeypatch.setattr(manager, "_find_pid_on_port", lambda port: {9000: 111, 9001: 222}.get(port))
    monkeypatch.setattr(DaemonEmbedManager, "_kill_process", staticmethod(lambda pid: killed.append(pid) or True))
    monkeypatch.setattr(manager, "_is_port_in_use", lambda port: False)

    # configured port is now 9001 (changed); recorded is still 9000
    assert manager.stop_ui("", ui_port=9001) is True
    assert sorted(killed) == [111, 222]  # both old and new killed
    assert not manager._ui_port_file(paths).exists()


def test_register_profile_preserves_existing_embed_keys(tmp_path, monkeypatch):
    """_register_profile rewrites the .env on daemon start; it must merge the
    existing non-API keys (UI port, idle timeout, ...) forward instead of
    dropping them. Regression for the HINDSIGHT_EMBED_CP_PORT wipe."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    manager = DaemonEmbedManager()
    # Seed a profile .env that already carries an embed-only key.
    manager._profile_manager.create_profile(
        "p", {"HINDSIGHT_API_LLM_PROVIDER": "openai", "HINDSIGHT_EMBED_CP_PORT": "25000"}
    )

    # Daemon start passes only HINDSIGHT_API_* config to _register_profile.
    manager._register_profile("p", 9100, {"HINDSIGHT_API_LLM_PROVIDER": "openai", "HINDSIGHT_API_LLM_API_KEY": "sk-x"})

    env = (tmp_path / ".hindsight" / "profiles" / "p.env").read_text()
    assert "HINDSIGHT_EMBED_CP_PORT=25000" in env  # preserved, not wiped
    assert "HINDSIGHT_API_LLM_API_KEY=sk-x" in env


def test_component_version_resolution(tmp_path, monkeypatch):
    """Component version: profile .env override > env var > embed __version__."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    from hindsight_embed import __version__

    manager = DaemonEmbedManager()
    manager._profile_manager.create_profile("p", {"HINDSIGHT_API_LLM_PROVIDER": "openai"})

    # default = embed version
    monkeypatch.delenv("HINDSIGHT_EMBED_CP_VERSION", raising=False)
    assert manager._component_version("p", "HINDSIGHT_EMBED_CP_VERSION") == __version__

    # env var overrides the default
    monkeypatch.setenv("HINDSIGHT_EMBED_CP_VERSION", "9.9.9")
    assert manager._component_version("p", "HINDSIGHT_EMBED_CP_VERSION") == "9.9.9"

    # profile .env override beats the env var
    manager._profile_manager.create_profile(
        "p", {"HINDSIGHT_API_LLM_PROVIDER": "openai", "HINDSIGHT_EMBED_CP_VERSION": "1.2.3"}
    )
    assert manager._component_version("p", "HINDSIGHT_EMBED_CP_VERSION") == "1.2.3"
