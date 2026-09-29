"""Tests for lib/bank.py project and bank name resolution."""

from unittest.mock import MagicMock, patch

from lib.bank import _resolve_project_name


def _cfg(**overrides):
    config = {"resolveWorktrees": True}
    config.update(overrides)
    return config


def _git_result(stdout, returncode=0):
    result = MagicMock()
    result.stdout = stdout
    result.returncode = returncode
    return result


@patch("lib.bank.subprocess.run")
def test_bare_hub_resolves_to_hidden_directory_parent(mock_run):
    mock_run.side_effect = [_git_result("/home/user/myrepo/.bare\n"), _git_result("true\n")]
    assert _resolve_project_name("/home/user/myrepo/main", _cfg()) == "myrepo"


@patch("lib.bank.subprocess.run")
def test_standalone_bare_keeps_common_directory_name(mock_run):
    mock_run.return_value = _git_result("/srv/repos/myproject.git\n")
    assert _resolve_project_name("/srv/repos/myproject.git", _cfg()) == "myproject.git"
