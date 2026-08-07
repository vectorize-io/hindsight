"""Tests for lib/bank.py — bank ID derivation and mission management."""

import json
import ntpath
import subprocess
from unittest.mock import MagicMock, patch

import pytest
from lib.bank import _resolve_project_name, derive_bank_id, ensure_bank_mission


def _cfg(**overrides):
    base = {
        "dynamicBankId": False,
        "bankId": "claude-code",
        "bankIdPrefix": "",
        "agentName": "claude-code",
        "dynamicBankGranularity": ["agent", "project"],
        "bankMission": "",
        "retainMission": None,
        "resolveWorktrees": True,
        "directoryBankMap": {},
    }
    base.update(overrides)
    return base


def _hook(session_id="sess-1", cwd="/home/user/myproject"):
    return {"session_id": session_id, "cwd": cwd}


class TestDeriveBankIdStatic:
    def test_static_default_bank(self):
        assert derive_bank_id(_hook(), _cfg()) == "claude-code"

    def test_static_custom_bank_id(self):
        cfg = _cfg(bankId="my-agent")
        assert derive_bank_id(_hook(), cfg) == "my-agent"

    def test_static_with_prefix(self):
        cfg = _cfg(bankId="bot", bankIdPrefix="prod")
        assert derive_bank_id(_hook(), cfg) == "prod-bot"

    def test_static_prefix_without_bankid_uses_default(self):
        cfg = _cfg(bankId=None, bankIdPrefix="dev")
        assert derive_bank_id(_hook(), cfg) == "dev-claude-code"


class TestDeriveBankIdDynamic:
    def test_dynamic_agent_project(self):
        cfg = _cfg(dynamicBankId=True, agentName="mybot", dynamicBankGranularity=["agent", "project"])
        result = derive_bank_id(_hook(cwd="/home/user/hindsight"), cfg)
        assert result == "mybot::hindsight"

    def test_dynamic_preserves_raw_special_chars(self):
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["project"])
        result = derive_bank_id(_hook(cwd="/home/user/my project"), cfg)
        assert "my project" in result
        assert "%" not in result

    def test_dynamic_preserves_raw_utf8(self):
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["project"])
        result = derive_bank_id(_hook(cwd="/home/user/мой проект"), cfg)
        assert "мой проект" in result
        assert "%" not in result

    def test_dynamic_session_field(self):
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["session"])
        result = derive_bank_id(_hook(session_id="abc-123"), cfg)
        assert "abc-123" in result

    def test_dynamic_with_prefix(self):
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["agent"], bankIdPrefix="v2")
        result = derive_bank_id(_hook(), cfg)
        assert result.startswith("v2-")

    def test_dynamic_channel_from_env(self, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_CHANNEL_ID", "telegram-123")
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["channel"])
        result = derive_bank_id(_hook(), cfg)
        assert "telegram-123" in result

    def test_dynamic_user_from_env(self, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_USER_ID", "user-456")
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["user"])
        result = derive_bank_id(_hook(), cfg)
        assert "user-456" in result

    def test_dynamic_missing_env_uses_defaults(self, monkeypatch):
        monkeypatch.delenv("HINDSIGHT_CHANNEL_ID", raising=False)
        monkeypatch.delenv("HINDSIGHT_USER_ID", raising=False)
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["channel", "user"])
        result = derive_bank_id(_hook(), cfg)
        assert "default" in result
        assert "anonymous" in result

    def test_dynamic_empty_cwd_uses_unknown(self):
        cfg = _cfg(dynamicBankId=True, dynamicBankGranularity=["project"])
        result = derive_bank_id({"session_id": "s", "cwd": ""}, cfg)
        assert "unknown" in result

    @patch("lib.bank.subprocess.run")
    def test_dynamic_worktree_resolves_to_main_repo(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "/home/user/myproject/.git\n"
        mock_run.return_value = mock_result

        cfg = _cfg(dynamicBankId=True, agentName="bot", dynamicBankGranularity=["agent", "project"])
        # Working in a worktree, but should resolve to the main repo name
        result = derive_bank_id(_hook(cwd="/home/user/myproject-wt1"), cfg)
        assert result == "bot::myproject"


class TestResolveProjectName:
    """Tests for git worktree resolution in project name derivation."""

    def _mock_git(self, stdout, returncode=0):
        """Create a mock for subprocess.run that simulates git output."""
        result = MagicMock()
        result.returncode = returncode
        result.stdout = stdout
        return result

    @patch("lib.bank.subprocess.run")
    def test_regular_repo(self, mock_run):
        mock_run.return_value = self._mock_git("/home/user/myproject/.git\n")
        assert _resolve_project_name("/home/user/myproject", _cfg()) == "myproject"

    @patch("lib.bank.subprocess.run")
    def test_worktree_resolves_to_main_repo(self, mock_run):
        # Worktree at /home/user/myproject-wt1, main repo at /home/user/myproject
        mock_run.return_value = self._mock_git("/home/user/myproject/.git\n")
        assert _resolve_project_name("/home/user/myproject-wt1", _cfg()) == "myproject"

    @patch("lib.bank.subprocess.run")
    def test_worktree_different_location(self, mock_run):
        # Worktree at /tmp/worktrees/feature-x, main repo at /home/user/hindsight
        mock_run.return_value = self._mock_git("/home/user/hindsight/.git\n")
        assert _resolve_project_name("/tmp/worktrees/feature-x", _cfg()) == "hindsight"

    @patch("lib.bank.subprocess.run")
    def test_disabled_falls_back_to_basename(self, mock_run):
        cfg = _cfg(resolveWorktrees=False)
        assert _resolve_project_name("/home/user/myproject-wt1", cfg) == "myproject-wt1"
        mock_run.assert_not_called()

    @patch("lib.bank.subprocess.run")
    def test_git_not_available(self, mock_run):
        mock_run.side_effect = OSError("git not found")
        assert _resolve_project_name("/home/user/myproject", _cfg()) == "myproject"

    @patch("lib.bank.subprocess.run")
    def test_not_a_git_repo(self, mock_run):
        mock_run.return_value = self._mock_git("", returncode=128)
        assert _resolve_project_name("/home/user/plaindir", _cfg()) == "plaindir"

    @patch("lib.bank.subprocess.run")
    def test_git_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=5)
        assert _resolve_project_name("/home/user/myproject", _cfg()) == "myproject"

    def test_empty_cwd(self):
        assert _resolve_project_name("", _cfg()) == "unknown"


class TestDirectoryBankMap:
    """Tests for explicit directory-to-bank mapping."""

    def test_exact_match(self):
        cfg = _cfg(directoryBankMap={"/home/user/myproject": "custom-bank"})
        result = derive_bank_id(_hook(cwd="/home/user/myproject"), cfg)
        assert result == "custom-bank"

    def test_match_with_trailing_slash(self):
        cfg = _cfg(directoryBankMap={"/home/user/myproject/": "custom-bank"})
        result = derive_bank_id(_hook(cwd="/home/user/myproject"), cfg)
        assert result == "custom-bank"

    def test_descendant_matches_configured_project_root(self):
        cfg = _cfg(directoryBankMap={"/home/user/myproject": "custom-bank"})
        result = derive_bank_id(_hook(cwd="/home/user/myproject/src/package"), cfg)
        assert result == "custom-bank"

    def test_nearest_configured_ancestor_wins(self):
        cfg = _cfg(
            directoryBankMap={
                "/home/user/myproject": "project-bank",
                "/home/user/myproject/packages/api": "api-bank",
            }
        )
        result = derive_bank_id(_hook(cwd="/home/user/myproject/packages/api/src"), cfg)
        assert result == "api-bank"

    def test_path_prefix_without_ancestor_boundary_does_not_match(self):
        cfg = _cfg(
            directoryBankMap={"/home/user/project": "project-bank"},
            bankId="fallback",
        )
        result = derive_bank_id(_hook(cwd="/home/user/project-other/src"), cfg)
        assert result == "fallback"

    @pytest.mark.skipif(not hasattr(__import__("os"), "symlink"), reason="symlinks not supported")
    def test_symlinked_descendant_matches_real_project_root(self, tmp_path):
        import os

        real_project = tmp_path / "project"
        nested_dir = real_project / "src"
        nested_dir.mkdir(parents=True)
        project_link = tmp_path / "project-link"
        os.symlink(real_project, project_link)

        cfg = _cfg(directoryBankMap={str(real_project): "project-bank"})
        result = derive_bank_id(_hook(cwd=str(project_link / "src")), cfg)
        assert result == "project-bank"

    @pytest.mark.skipif(not hasattr(__import__("os"), "symlink"), reason="symlinks not supported")
    def test_symlinked_mapping_does_not_capture_canonical_sibling(self, tmp_path):
        import os

        workspace = tmp_path / "workspace"
        project_a = workspace / "project-a"
        project_b = workspace / "project-b" / "src"
        project_a.mkdir(parents=True)
        project_b.mkdir(parents=True)
        workspace_link = project_a / "workspace-link"
        os.symlink(workspace, workspace_link)

        cfg = _cfg(directoryBankMap={str(workspace_link): "project-a-bank"}, bankId="fallback")
        assert derive_bank_id(_hook(cwd=str(workspace_link / "project-b" / "src")), cfg) == "project-a-bank"
        result = derive_bank_id(_hook(cwd=str(project_b)), cfg)
        assert result == "fallback"

    @pytest.mark.skipif(not hasattr(__import__("os"), "symlink"), reason="symlinks not supported")
    def test_nested_symlink_cannot_escape_mapped_tree(self, tmp_path):
        import os

        project = tmp_path / "project"
        outside = tmp_path / "outside" / "src"
        project.mkdir()
        outside.mkdir(parents=True)
        escape_link = project / "escape"
        os.symlink(outside.parent, escape_link)

        cfg = _cfg(directoryBankMap={str(project): "project-bank"}, bankId="fallback")
        assert derive_bank_id(_hook(cwd=str(escape_link / "src")), cfg) == "fallback"

    def test_parent_segments_cannot_escape_mapped_tree(self, tmp_path):
        project = tmp_path / "project"
        sibling = tmp_path / "sibling" / "src"
        project.mkdir()
        sibling.mkdir(parents=True)

        cfg = _cfg(directoryBankMap={str(project): "project-bank"}, bankId="fallback")
        escaped_cwd = str(project / ".." / "sibling" / "src")
        assert derive_bank_id(_hook(cwd=escaped_cwd), cfg) == "fallback"

    @pytest.mark.skipif(not hasattr(__import__("os"), "symlink"), reason="symlinks not supported")
    def test_nearest_ancestor_ignores_symlink_spelling_length_and_map_order(self, tmp_path):
        import os

        canonical_root = tmp_path / "r"
        canonical_project = canonical_root / "project"
        (canonical_project / "src").mkdir(parents=True)
        long_alias = tmp_path / "a-very-long-workspace-alias-that-must-not-win"
        os.symlink(canonical_root, long_alias)

        entries = [
            (str(long_alias), "root-bank"),
            (str(canonical_project), "project-bank"),
        ]
        cwd = str(long_alias / "project" / "src")
        for ordered_entries in (entries, list(reversed(entries))):
            cfg = _cfg(directoryBankMap=dict(ordered_entries))
            assert derive_bank_id(_hook(cwd=cwd), cfg) == "project-bank"

    def test_relative_mapping_keeps_exact_match_but_not_descendants(self, tmp_path, monkeypatch):
        project = tmp_path / "project"
        nested_dir = project / "src"
        nested_dir.mkdir(parents=True)
        monkeypatch.chdir(project)

        cfg = _cfg(directoryBankMap={".": "project-bank"}, bankId="fallback")
        assert derive_bank_id(_hook(cwd=str(project)), cfg) == "project-bank"
        assert derive_bank_id(_hook(cwd=str(nested_dir)), cfg) == "fallback"

    @pytest.mark.skipif(not hasattr(__import__("os"), "symlink"), reason="symlinks not supported")
    def test_conflicting_canonical_roots_fall_through_regardless_of_order(self, tmp_path, capsys):
        import os

        project = tmp_path / "project"
        project.mkdir()
        project_link = tmp_path / "project-link"
        os.symlink(project, project_link)

        entries = [(str(project), "bank-a"), (str(project_link), "bank-b")]
        for ordered_entries in (entries, list(reversed(entries))):
            cfg = _cfg(directoryBankMap=dict(ordered_entries), bankId="fallback")
            assert derive_bank_id(_hook(cwd=str(project)), cfg) == "fallback"

        assert capsys.readouterr().err.count("Conflicting directoryBankMap entries") == 2

    @pytest.mark.skipif(not hasattr(__import__("os"), "symlink"), reason="symlinks not supported")
    def test_conflicted_nearest_root_blocks_parent_but_not_deeper_mapping(self, tmp_path, capsys):
        import os

        projects = tmp_path / "projects"
        project = projects / "a"
        nested = project / "nested"
        child = nested / "src"
        child.mkdir(parents=True)
        project_link = tmp_path / "project-a-link"
        os.symlink(project, project_link)

        mappings = {
            str(projects): "parent-bank",
            str(project): "bank-a",
            str(project_link): "bank-b",
        }
        cfg = _cfg(directoryBankMap=mappings, bankId="fallback")
        assert derive_bank_id(_hook(cwd=str(project / "src")), cfg) == "fallback"

        mappings[str(nested)] = "nested-bank"
        cfg = _cfg(directoryBankMap=mappings, bankId="fallback")
        assert derive_bank_id(_hook(cwd=str(child)), cfg) == "nested-bank"
        assert capsys.readouterr().err.count("Conflicting directoryBankMap entries") == 1

    def test_mapping_realpath_is_snapshotted_once_regardless_of_order(self):
        entries = [("/alias", "alias-bank"), ("/project", "project-bank")]
        for ordered_entries in (entries, list(reversed(entries))):
            alias_calls = 0

            def changing_realpath(path):
                nonlocal alias_calls
                if path == "/alias":
                    alias_calls += 1
                    return "/elsewhere" if alias_calls == 1 else "/project"
                return path

            with patch("lib.bank.os.path.realpath", side_effect=changing_realpath):
                cfg = _cfg(directoryBankMap=dict(ordered_entries), bankId="fallback")
                result = derive_bank_id(_hook(cwd="/project/src"), cfg)
            assert result == "project-bank"
            assert alias_calls == 1

    @pytest.mark.parametrize(
        ("cwd", "dir_path", "expected"),
        [
            (r"C:\Project\src", r"c:\project", "windows-bank"),
            (r"C:\project-other\src", r"C:\project", "fallback"),
            (r"C:\project\src", r"D:\project", "fallback"),
            (r"\\server\share\project\src", r"\\server\share", "windows-bank"),
            (r"\\server\share\project\src", r"\\SERVER\SHARE\project", "windows-bank"),
            (r"\\server\other\project\src", r"\\server\share\project", "fallback"),
        ],
    )
    def test_windows_drive_case_and_unc_semantics_with_ntpath(self, cwd, dir_path, expected):
        with (
            patch("lib.bank.os.path.abspath", side_effect=ntpath.abspath),
            patch("lib.bank.os.path.isabs", side_effect=ntpath.isabs),
            patch("lib.bank.os.path.normcase", side_effect=ntpath.normcase),
            patch("lib.bank.os.path.realpath", side_effect=ntpath.realpath),
            patch("lib.bank.os.path.relpath", side_effect=ntpath.relpath),
            patch("lib.bank.os.sep", ntpath.sep),
        ):
            cfg = _cfg(directoryBankMap={dir_path: "windows-bank"}, bankId="fallback")
            assert derive_bank_id(_hook(cwd=cwd), cfg) == expected

    def test_windows_drive_letter_case_insensitive_match(self):
        # On Windows, the cwd's drive-letter (and path) case depends on the
        # launcher: PowerShell and git-bash hand children an UPPERCASE drive
        # while the VS Code extension spawn reports lowercase. The map match
        # must not depend on which launcher started the session. normcase is
        # a no-op on POSIX, so this test only exercises the Windows behavior
        # when run there; the POSIX-case-sensitivity test below pins the
        # complementary guarantee.
        import os as _os

        if _os.path.normcase("A") == _os.path.normcase("a"):  # case-insensitive FS semantics
            cfg = _cfg(directoryBankMap={r"c:\Users\dev\proj": "custom-bank"})
            result = derive_bank_id(_hook(cwd=r"C:\Users\dev\proj"), cfg)
            assert result == "custom-bank"

    def test_posix_paths_stay_case_sensitive(self):
        import os as _os

        if _os.path.normcase("A") != _os.path.normcase("a"):  # POSIX semantics
            cfg = _cfg(directoryBankMap={"/home/User/myproject": "custom-bank"}, bankId="default-bank")
            result = derive_bank_id(_hook(cwd="/home/user/myproject"), cfg)
            assert result == "default-bank"

    def test_no_match_falls_through_to_static(self):
        cfg = _cfg(directoryBankMap={"/home/user/other": "other-bank"}, bankId="default-bank")
        result = derive_bank_id(_hook(cwd="/home/user/myproject"), cfg)
        assert result == "default-bank"

    def test_no_match_falls_through_to_dynamic(self):
        cfg = _cfg(
            directoryBankMap={"/home/user/other": "other-bank"},
            dynamicBankId=True,
            agentName="bot",
            dynamicBankGranularity=["agent"],
            resolveWorktrees=False,
        )
        result = derive_bank_id(_hook(cwd="/home/user/myproject"), cfg)
        assert result == "bot"

    def test_with_prefix(self):
        cfg = _cfg(
            directoryBankMap={"/home/user/myproject": "custom-bank"},
            bankIdPrefix="prod",
        )
        result = derive_bank_id(_hook(cwd="/home/user/myproject"), cfg)
        assert result == "prod-custom-bank"

    def test_overrides_dynamic_mode(self):
        cfg = _cfg(
            directoryBankMap={"/home/user/myproject": "explicit-bank"},
            dynamicBankId=True,
            agentName="bot",
            dynamicBankGranularity=["agent", "project"],
        )
        result = derive_bank_id(_hook(cwd="/home/user/myproject"), cfg)
        assert result == "explicit-bank"

    def test_empty_map_ignored(self):
        cfg = _cfg(directoryBankMap={}, bankId="default-bank")
        result = derive_bank_id(_hook(), cfg)
        assert result == "default-bank"

    def test_empty_cwd_skips_map(self):
        cfg = _cfg(directoryBankMap={"/some/path": "mapped-bank"}, bankId="fallback")
        result = derive_bank_id({"session_id": "s", "cwd": ""}, cfg)
        assert result == "fallback"

    def test_multiple_entries(self):
        cfg = _cfg(
            directoryBankMap={
                "/home/user/project-a": "bank-a",
                "/home/user/project-b": "bank-b",
            }
        )
        assert derive_bank_id(_hook(cwd="/home/user/project-a"), cfg) == "bank-a"
        assert derive_bank_id(_hook(cwd="/home/user/project-b"), cfg) == "bank-b"


class TestEnsureBankMission:
    def test_sets_mission_on_first_call(self, state_dir):
        client = MagicMock()
        cfg = _cfg(bankMission="You are a helpful assistant.", bankId="test-bank")
        ensure_bank_mission(client, "test-bank", cfg)
        client.set_bank_mission.assert_called_once_with(
            "test-bank", "You are a helpful assistant.", retain_mission=None, timeout=10
        )

    def test_skips_if_already_set(self, state_dir):
        client = MagicMock()
        cfg = _cfg(bankMission="mission text")
        ensure_bank_mission(client, "bank-a", cfg)
        ensure_bank_mission(client, "bank-a", cfg)  # second call
        assert client.set_bank_mission.call_count == 1

    def test_skips_if_mission_empty(self, state_dir):
        client = MagicMock()
        cfg = _cfg(bankMission="")
        ensure_bank_mission(client, "bank-b", cfg)
        client.set_bank_mission.assert_not_called()

    def test_includes_retain_mission_if_set(self, state_dir):
        client = MagicMock()
        cfg = _cfg(bankMission="reflect mission", retainMission="retain mission")
        ensure_bank_mission(client, "bank-c", cfg)
        client.set_bank_mission.assert_called_once_with(
            "bank-c", "reflect mission", retain_mission="retain mission", timeout=10
        )

    def test_graceful_on_api_error(self, state_dir):
        client = MagicMock()
        client.set_bank_mission.side_effect = RuntimeError("server down")
        cfg = _cfg(bankMission="mission")
        # Should not raise
        ensure_bank_mission(client, "bank-d", cfg)

    def test_different_banks_each_set_once(self, state_dir):
        client = MagicMock()
        cfg = _cfg(bankMission="mission")
        ensure_bank_mission(client, "bank-x", cfg)
        ensure_bank_mission(client, "bank-y", cfg)
        assert client.set_bank_mission.call_count == 2

    @pytest.mark.skipif(not hasattr(__import__("os"), "symlink"), reason="symlinks not supported")
    def test_directorybankmap_matches_symlinked_cwd(self, tmp_path):
        import os

        real = os.path.realpath(tmp_path / "proj")
        os.makedirs(real)
        link = str(tmp_path / "proj-link")
        os.symlink(real, link)
        cfg = _cfg(directoryBankMap={real: "myproj"}, bankId="fallback")
        assert derive_bank_id({"cwd": real, "session_id": "s"}, cfg) == "myproj"
        assert derive_bank_id({"cwd": link, "session_id": "s"}, cfg) == "myproj"
