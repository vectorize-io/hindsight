"""Project routing is opt-in, worktree-aware, and never guesses after probe errors."""

import errno
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from conftest import plugin
from hindsight_hermes import project


def git(path, *args):
    subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)


def repository(path, *, bare=False):
    path.mkdir()
    git(path, "init", *(["--bare"] if bare else []))
    if not bare:
        git(
            path,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-m",
            "init",
        )
    return path


@pytest.mark.parametrize("name", ["main", "main.git"])
def test_common_identity_real_worktrees(tmp_path, monkeypatch, name):
    main = repository(tmp_path / name)
    worktree = tmp_path / "feature"
    git(main, "worktree", "add", str(worktree))
    nested = worktree / "nested"
    nested.mkdir()
    monkeypatch.setenv("GIT_DIR", "/wrong/repository")
    monkeypatch.setenv("GIT_COMMON_DIR", "/wrong/common")
    monkeypatch.setattr(subprocess, "run", Mock(side_effect=AssertionError("no subprocess probe")))
    for cwd in (main, worktree, nested):
        assert project.resolve_git_project(str(cwd)) == name
        assert project.probe_git_layout(str(cwd)).status == "resolved"


@pytest.mark.parametrize("core_header, bare_value", [("[core]", "true"), ("[core] # comment", "yes # comment")])
def test_bare_hub_and_standalone(tmp_path, core_header, bare_value):
    bare = repository(tmp_path / "standalone.git", bare=True)
    assert project.resolve_git_project(str(bare)) == "standalone.git"
    hub = tmp_path / "hub"
    hub.mkdir()
    repository(hub / ".bare", bare=True)
    (hub / ".bare" / "config").write_text(f"{core_header}\n bare = {bare_value}\n")
    (hub / ".git").write_text("gitdir: ./.bare\n")
    assert project.resolve_git_project(str(hub)) == "hub"


def test_nonrepo_basename(tmp_path, monkeypatch):
    # Test runners may keep scratch inside a versioned home. Hide only ancestor
    # markers to model a real nonrepo, while exercising the complete root walk.
    original = project._entry_kind
    ancestor_markers = {parent / ".git" for parent in tmp_path.parents}
    monkeypatch.setattr(project, "_entry_kind", lambda path: "absent" if path in ancestor_markers else original(path))
    assert project.probe_git_layout(str(tmp_path)).status == "absent"
    assert project.resolve_git_project(str(tmp_path)) == tmp_path.name


@pytest.mark.parametrize("pointer", ["", "not git", "gitdir: missing", "gitdir: \n", "gitdir: /one\nother"])
def test_bad_git_pointer_fails_closed(tmp_path, pointer):
    (tmp_path / ".git").write_text(pointer)
    assert project.probe_git_layout(str(tmp_path)).status == "failed"
    with pytest.raises(project.GitProjectResolutionError):
        project.resolve_git_project(str(tmp_path))


@pytest.mark.parametrize("pointer", ["", "missing", "../one\n../two"])
def test_bad_common_pointer_fails_closed(tmp_path, pointer):
    main = repository(tmp_path / "main")
    (main / ".git" / "commondir").write_text(pointer)
    assert project.probe_git_layout(str(main)).status == "failed"


@pytest.mark.parametrize(
    "error", [PermissionError(errno.EACCES, "denied"), OSError(errno.EIO, "IO"), TimeoutError("timeout")]
)
def test_io_failure_is_not_absence(tmp_path, monkeypatch, error):
    monkeypatch.setattr(Path, "lstat", Mock(side_effect=error))
    assert project.probe_git_layout(str(tmp_path)).status == "failed"
    with pytest.raises(project.GitProjectResolutionError):
        project.resolve_git_project(str(tmp_path))


def test_missing_cwd_is_failure(tmp_path):
    assert project.probe_git_layout(str(tmp_path / "deleted-worktree")).status == "failed"


def test_explicit_and_remote_do_not_probe(monkeypatch):
    monkeypatch.setattr(project, "probe_git_layout", Mock(side_effect=AssertionError("no host IO")))
    assert project.resolve_git_project("/remote/path", "chosen", backend="ssh") == "chosen"
    assert project.resolve_git_project("/remote/path", backend="ssh") == "path"
    assert project.resolve_git_project("C:\\remote\\path", backend="ssh") == "path"
    assert project.resolve_git_project("", backend="ssh") == ""
    assert project.resolve_git_project("/missing", "chosen") == "chosen"
    monkeypatch.setattr(project.os, "getcwd", Mock(side_effect=FileNotFoundError("deleted launch directory")))
    assert project.resolve_git_project("", "chosen") == "chosen"
    assert project.resolve_git_project("", backend="ssh") == ""


@pytest.mark.parametrize("template", ["", "bank-{profile}", "bank-{workspace}", "bank-{{gitProject}}"])
def test_legacy_templates_do_not_resolve_context(provider, monkeypatch, template):
    monkeypatch.setattr(plugin, "capture_project_context", Mock(side_effect=AssertionError("not opt-in")))
    instance, _ = provider({"bank_id_template": template}, agent_workspace="hermes")
    assert instance._agent_workspace == "hermes"
    instance.shutdown()


def test_project_frozen_and_workspace_unchanged(provider, monkeypatch, tmp_path):
    main = repository(tmp_path / "main")
    monkeypatch.setattr(project, "capture_project_context", lambda cwd="": project.ProjectContext(str(main), "local"))
    monkeypatch.setattr(plugin, "capture_project_context", project.capture_project_context)
    instance, fake = provider({"bank_id_template": "{workspace}-{gitProject}-{session}"}, agent_workspace="hermes")
    instance.sync_turn("one", "1")
    monkeypatch.setattr(
        plugin, "capture_project_context", Mock(side_effect=AssertionError("identity must stay frozen"))
    )
    instance.on_session_switch("session-2")
    instance.sync_turn("two", "2")
    instance.handle_tool_call("hindsight_recall", {"query": "where?"})
    instance.handle_tool_call("hindsight_reflect", {"query": "why?"})
    instance.shutdown()
    assert instance._agent_workspace == "hermes"
    assert [call["bank_id"] for call in fake.retains] == ["hermes-main-session-1", "hermes-main-session-2"]
    assert fake.recalls[0]["bank_id"] == "hermes-main-session-2"
    assert fake.reflects[0]["bank_id"] == "hermes-main-session-2"


def test_queued_retains_and_operations_keep_original_banks(provider, monkeypatch):
    instance, fake = provider({"bank_id_template": "bank-{session}"})
    jobs = []
    monkeypatch.setattr(instance, "_enqueue_retain", jobs.append)
    instance.sync_turn("one", "1")
    instance.on_session_switch("session-2")
    instance.sync_turn("two", "2")
    for job in jobs:
        job()
    assert fake.retains[0]["bank_id"] == "bank-session-1"
    assert fake.retains[-1]["bank_id"] == "bank-session-2"
    for bank in ("bank-session-1", "bank-session-2"):
        instance._track_retain_ops(SimpleNamespace(operation_id="same-id"), bank)
    checked = []
    monkeypatch.setattr(instance, "_is_retain_op_complete", lambda bank, op: checked.append((bank, op)) or True)
    assert instance._wait_for_server_retain_ops(lambda: False, 1)
    assert set(checked) == {("bank-session-1", "same-id"), ("bank-session-2", "same-id")}
    instance.shutdown()


@pytest.mark.parametrize("template", ["bank-{gitProject}", "bank-{gitProject!s}", "bank-{gitProject:.4}"])
def test_formatted_project_placeholder_is_lazy_opt_in(provider, monkeypatch, template):
    capture = Mock(return_value=project.ProjectContext("/remote/project", "ssh"))
    monkeypatch.setattr(plugin, "capture_project_context", capture)
    instance, _ = provider({"bank_id_template": template, "git_project": "explicit"})
    capture.assert_called_once()
    assert instance._bank_id == ("bank-expl" if ":.4" in template else "bank-explicit")
    instance.shutdown()


def test_template_fields_nested_and_escaped():
    assert plugin._template_fields("{{gitProject}}-{profile}") == {"profile"}
    assert plugin._template_fields("{profile:{gitProject}}") == {"profile", "gitProject"}
    assert plugin._resolve_bank_id_template("{broken", "fallback") == "fallback"


@pytest.mark.parametrize("entry", [".git", ".git/config", ".git/commondir"])
def test_unreadable_layout_never_walks_to_parent_repo(tmp_path, monkeypatch, entry):
    main = repository(tmp_path / "main")
    original = Path.read_text
    target = main / entry
    if entry == ".git":
        worktree = tmp_path / "feature"
        git(main, "worktree", "add", str(worktree))
        target = worktree / ".git"
        cwd = worktree
    else:
        cwd = main
        if entry.endswith("commondir"):
            target.write_text(".\n")

    def read(path, *args, **kwargs):
        if path == target:
            raise PermissionError(errno.EACCES, "denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    assert project.probe_git_layout(str(cwd)).status == "failed"


def test_transient_read_retries_same_identity(tmp_path, monkeypatch):
    main = repository(tmp_path / "main")
    original = project._probe_once
    calls = []

    def probe(cwd):
        calls.append(cwd)
        if len(calls) < 3:
            raise OSError(errno.EAGAIN, "resource pressure")
        return original(cwd)

    monkeypatch.setattr(project, "_probe_once", probe)
    assert project.resolve_git_project(str(main)) == "main"
    assert calls == [str(main)] * 3


def test_home_repository_is_not_excluded(tmp_path, monkeypatch):
    home = repository(tmp_path / "home")
    monkeypatch.setenv("HOME", str(home))
    assert project.resolve_git_project(str(home)) == "home"


@pytest.mark.skipif(sys.platform == "win32", reason="Symlinks require elevated privileges on Windows")
def test_dangling_marker_fails(tmp_path):
    (tmp_path / ".git").symlink_to(tmp_path / "missing")
    assert project.probe_git_layout(str(tmp_path)).status == "failed"


def test_nested_repository_uses_nearest_identity(tmp_path):
    outer = repository(tmp_path / "outer")
    inner = repository(outer / "inner")
    assert project.resolve_git_project(str(inner)) == "inner"


def test_empty_project_uses_configured_static_fallback(provider, monkeypatch):
    monkeypatch.setattr(plugin, "capture_project_context", lambda cwd: project.ProjectContext("", "ssh"))
    instance, _ = provider({"bank_id_template": "project::{gitProject}", "bank_id": "safe-static"})
    assert instance._bank_id == "safe-static"
    instance.on_session_switch("new")
    assert instance._bank_id == "safe-static"
    instance.shutdown()
