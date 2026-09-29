"""Tests for per-project bank derivation."""

import subprocess

import pytest

from hindsight_devin_desktop.project import derive_project_slug, project_bank_id, slug_from_remote


class TestSlugFromRemote:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("git@github.com:acme/web.git", "acme-web"),
            ("https://github.com/acme/web.git", "acme-web"),
            ("https://github.com/acme/web", "acme-web"),
            ("ssh://git@github.com/acme/web.git", "acme-web"),
            ("git@gitlab.com:group/sub/proj.git", "group-sub-proj"),
            ("https://user:token@github.com/Acme/My_Repo.git", "acme-my-repo"),
        ],
    )
    def test_owner_repo_slug(self, url, expected):
        assert slug_from_remote(url) == expected

    def test_same_slug_across_protocols(self):
        # Teammates cloning over different protocols get the same project bank.
        assert slug_from_remote("git@github.com:acme/web.git") == slug_from_remote("https://github.com/acme/web.git")


def _git(args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


class TestDerive:
    def test_prefers_git_remote(self, tmp_path):
        _git(["init", "-q"], tmp_path)
        _git(["remote", "add", "origin", "git@github.com:acme/web.git"], tmp_path)
        slug, source = derive_project_slug(tmp_path)
        assert slug == "acme-web"
        assert "remote" in source

    def test_falls_back_to_repo_folder(self, tmp_path):
        repo = tmp_path / "MyCoolRepo"
        repo.mkdir()
        _git(["init", "-q"], repo)  # no remote
        slug, source = derive_project_slug(repo)
        assert slug == "mycoolrepo"
        assert "folder" in source

    def test_falls_back_to_folder_without_git(self, tmp_path):
        plain = tmp_path / "Plain Dir"
        plain.mkdir()
        slug, source = derive_project_slug(plain)
        assert slug == "plain-dir"

    def test_project_bank_id_prefixes_global(self, tmp_path):
        _git(["init", "-q"], tmp_path)
        _git(["remote", "add", "origin", "git@github.com:acme/web.git"], tmp_path)
        bank, _ = project_bank_id("devin-desktop", tmp_path)
        assert bank == "devin-desktop-acme-web"
