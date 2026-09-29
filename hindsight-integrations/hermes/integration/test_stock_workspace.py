"""Run separately from tests/ against real stock Hermes (see README Development).

No Hermes interface stubs: real caller kwargs, profile scope and MemoryManager.
Only Hindsight network/client operations and terminal launching are blocked.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from agent.agent_init import _GATEWAY_IDENTITY_PARAMS, _memory_provider_init_kwargs
from agent.memory_manager import MemoryManager
from agent.runtime_cwd import reset_session_cwd, set_session_cwd
from gateway.run import _profile_runtime_scope
from tools.terminal_scope import TerminalPolicyUnavailable

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "hindsight_stock_test", PLUGIN_ROOT / "__init__.py", submodule_search_locations=[str(PLUGIN_ROOT)]
)
assert spec is not None and spec.loader is not None
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)


def init_kwargs(cwd: str | None = None):
    agent = SimpleNamespace(
        session_id="stock-session",
        session_cwd=cwd,
        _session_db=None,
        _emit_warning=lambda *a: None,
        _emit_status=lambda *a: None,
        **{"_" + key: None for key in _GATEWAY_IDENTITY_PARAMS},
    )
    return _memory_provider_init_kwargs(agent, "telegram")


def profile(home: Path, backend: str, cwd: str, **config) -> Path:
    home.mkdir()
    (home / "config.yaml").write_text(f"terminal:\n  backend: {backend}\n  cwd: {json.dumps(cwd)}\n")
    (home / "hindsight").mkdir()
    (home / "hindsight" / "config.json").write_text(
        json.dumps({"mode": "cloud", "api_key": "test-only", "bank_id_template": "{workspace}-{gitProject}", **config})
    )
    return home


@pytest.fixture(autouse=True)
def no_external_work(monkeypatch):
    import tools.terminal_tool_backends

    monkeypatch.setattr(
        tools.terminal_tool_backends, "_create_environment", Mock(side_effect=AssertionError("no backend launch"))
    )
    monkeypatch.setattr(plugin, "_warn_if_client_outdated", lambda: None)
    monkeypatch.setattr(plugin, "_check_api_supports_update_mode_append", lambda *a, **kw: True)
    monkeypatch.setattr(
        plugin.HindsightMemoryProvider, "_new_cloud_client", Mock(side_effect=AssertionError("no network"))
    )
    monkeypatch.setattr(
        plugin.HindsightMemoryProvider, "_start_embedded_daemon", Mock(side_effect=AssertionError("no daemon"))
    )
    yield
    tools.terminal_tool_backends._create_environment.assert_not_called()
    plugin.HindsightMemoryProvider._new_cloud_client.assert_not_called()
    plugin.HindsightMemoryProvider._start_embedded_daemon.assert_not_called()


def test_stock_aba_scoped_profiles_not_ambient_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", "/wrong/launch-profile")
    a = profile(tmp_path / "a", "ssh", "/remote-only/project-a")
    b = profile(tmp_path / "b", "docker", "/container-only/project-b")
    # Remote routing must not inspect even one host .git marker.
    monkeypatch.setattr(plugin.project, "probe_git_layout", Mock(side_effect=AssertionError("remote host probe")))
    seen = []
    for home, expected in ((a, "hermes-project-a"), (b, "hermes-project-b"), (a, "hermes-project-a")):
        with _profile_runtime_scope(home, prepared_secret_scope={}):
            kwargs = init_kwargs()
            assert "cwd" not in kwargs and "workspace_backend" not in kwargs
            assert kwargs["agent_workspace"] == "hermes"
            instance = plugin.HindsightMemoryProvider()
            manager = MemoryManager()
            manager.add_provider(instance)
            manager.initialize_all(**kwargs)
            assert instance in manager.providers
            assert instance._agent_workspace == "hermes"
            assert instance._bank_id == expected
            seen.append(instance._bank_id)
            instance.shutdown()
    assert seen == ["hermes-project-a", "hermes-project-b", "hermes-project-a"]


def test_stock_explicit_session_terminal_precedence(tmp_path):
    home = profile(tmp_path / "a", "ssh", "/remote/terminal")
    with _profile_runtime_scope(home, prepared_secret_scope={}):
        token = set_session_cwd("/remote/session")
        try:
            for cwd, expected in ((None, "session"), ("/remote/explicit", "explicit")):
                instance = plugin.HindsightMemoryProvider()
                instance.initialize(**init_kwargs(cwd))
                assert instance._bank_id == f"hermes-{expected}"
                instance.shutdown()
        finally:
            reset_session_cwd(token)
        instance = plugin.HindsightMemoryProvider()
        instance.initialize(**init_kwargs())
        assert instance._bank_id == "hermes-terminal"
        instance.shutdown()


@pytest.mark.parametrize("override", ["", "chosen"])
def test_stock_refusal_is_not_bypassed_by_explicit_cwd_or_override(tmp_path, monkeypatch, override):
    home = profile(tmp_path / "broken", "ssh", "/remote/default", git_project=override)
    (home / "config.yaml").write_text("terminal: [invalid")
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    with _profile_runtime_scope(home, prepared_secret_scope={}):
        instance = plugin.HindsightMemoryProvider()
        with pytest.raises(TerminalPolicyUnavailable):
            instance.initialize(**init_kwargs(str(tmp_path)))
        # Stock MemoryManager keeps failed providers registered; the plugin must
        # latch the refusal so tools/hooks cannot fall back to its default bank.
        manager = MemoryManager()
        manager.add_provider(instance)
        manager.initialize_all(**init_kwargs(str(tmp_path)))
        assert instance in manager.providers
        assert instance._bank_resolution_error
        instance.sync_turn("must not write", "anything")
        assert not instance._session_turns
        assert instance._recall_disabled()
        assert instance.prefetch("must not recall") == ""
        instance.queue_prefetch("must not recall")
        assert instance.get_tool_schemas() == []
        assert instance.system_prompt_block() == ""
        with pytest.raises(RuntimeError, match="project routing unavailable"):
            instance._get_client()
        for name, args in (
            ("hindsight_retain", {"content": "must not write"}),
            ("hindsight_recall", {"query": "must not read"}),
            ("hindsight_reflect", {"query": "must not read"}),
        ):
            response = instance.handle_tool_call(name, args)
            assert "project routing unavailable" in response
        instance.shutdown()


def test_stock_local_worktree_cwd_and_legacy_workspace(tmp_path):
    import subprocess

    main = tmp_path / "main"
    main.mkdir()
    for args in (
        ["init"],
        ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--allow-empty", "-m", "init"],
        ["worktree", "add", str(tmp_path / "feature")],
    ):
        subprocess.run(["git", "-C", str(main), *args], check=True, capture_output=True)
    home = profile(tmp_path / "local", "local", str(tmp_path / "feature"))
    with _profile_runtime_scope(home, prepared_secret_scope={}):
        instance = plugin.HindsightMemoryProvider()
        manager = MemoryManager()
        manager.add_provider(instance)
        manager.initialize_all(**init_kwargs())
        assert instance in manager.providers
        assert instance._bank_id == "hermes-main"
        assert instance._agent_workspace == "hermes"
        instance.shutdown()


def test_stock_remote_unknown_cwd_uses_static_bank(tmp_path):
    home = profile(tmp_path / "remote", "ssh", "", bank_id="explicit-fallback")
    with _profile_runtime_scope(home, prepared_secret_scope={}):
        instance = plugin.HindsightMemoryProvider()
        instance.initialize(**init_kwargs())
        assert instance._bank_id == "explicit-fallback"
        instance.shutdown()


def test_stock_local_detection_failure_latches_refusal(tmp_path):
    broken = tmp_path / "broken-repo"
    broken.mkdir()
    (broken / ".git").write_text("gitdir: missing-administrative-directory\n")
    home = profile(tmp_path / "local", "local", str(broken), bank_id="must-not-fallback")
    with _profile_runtime_scope(home, prepared_secret_scope={}):
        instance = plugin.HindsightMemoryProvider()
        manager = MemoryManager()
        manager.add_provider(instance)
        manager.initialize_all(**init_kwargs())
        assert instance._bank_resolution_error
        assert instance.get_tool_schemas() == []
        instance.sync_turn("private", "do not store in a guessed bank")
        assert not instance._session_turns
        assert instance.prefetch("private") == ""
        instance.queue_prefetch("private")
        assert "project routing unavailable" in instance.handle_tool_call("hindsight_retain", {"content": "private"})
        with pytest.raises(RuntimeError, match="project routing unavailable"):
            instance._get_client()
        instance.shutdown()
