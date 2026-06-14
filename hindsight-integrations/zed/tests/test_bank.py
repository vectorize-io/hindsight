"""Tests for per-project bank derivation."""

import subprocess

from hindsight_zed.bank import bank_id_for_project, bank_id_for_thread_paths, project_name
from hindsight_zed.config import ZedConfig


def _git_init(path):
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)


def test_project_name_from_git_root(tmp_path):
    repo = tmp_path / "My_Cool.Repo"
    repo.mkdir()
    _git_init(repo)
    sub = repo / "src" / "deep"
    sub.mkdir(parents=True)
    # A nested dir resolves to the repo root basename, slugified.
    assert project_name(str(sub)) == "my_cool.repo"


def test_project_name_non_git_uses_basename(tmp_path):
    d = tmp_path / "Plain Folder"
    d.mkdir()
    assert project_name(str(d)) == "plain-folder"


def test_bank_id_per_project(tmp_path):
    d = tmp_path / "acme"
    d.mkdir()
    cfg = ZedConfig(bank_prefix="zed")
    assert bank_id_for_project(str(d), cfg) == "zed-acme"


def test_bank_id_fixed_overrides(tmp_path):
    d = tmp_path / "acme"
    d.mkdir()
    cfg = ZedConfig(fixed_bank_id="shared")
    assert bank_id_for_project(str(d), cfg) == "shared"


def test_bank_id_for_thread_paths_uses_first_existing(tmp_path):
    real = tmp_path / "real-proj"
    real.mkdir()
    cfg = ZedConfig(bank_prefix="zed")
    bid = bank_id_for_thread_paths(["/does/not/exist", str(real)], cfg)
    assert bid == "zed-real-proj"


def test_bank_id_for_thread_paths_none_when_empty():
    assert bank_id_for_thread_paths([], ZedConfig()) is None
