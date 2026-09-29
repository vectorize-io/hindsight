"""Tests for bank resolution (per git repo)."""

import subprocess

from hindsight_aider.bank import resolve_bank_id
from hindsight_aider.config import AiderConfig


def test_explicit_bank_id_wins(tmp_path):
    cfg = AiderConfig(bank_id="My Project!")
    assert resolve_bank_id(cfg, cwd=tmp_path) == "my-project"  # sanitized


def test_derives_from_git_repo_name(tmp_path):
    repo = tmp_path / "Cool_Repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    cfg = AiderConfig()  # no explicit bank
    assert resolve_bank_id(cfg, cwd=repo) == "cool_repo"


def test_non_git_falls_back_to_dir_name(tmp_path):
    d = tmp_path / "loose-dir"
    d.mkdir()
    cfg = AiderConfig()
    assert resolve_bank_id(cfg, cwd=d) == "loose-dir"
