"""Hermetic launcher discovery tests; never start a real Hindsight daemon."""

import os
import sys
from types import SimpleNamespace

import pytest

from hindsight_hermes import embedded

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX managed uv layout")


def test_managed_uvx_is_available_to_the_real_helper(monkeypatch, tmp_path):
    tools = tmp_path / ".hermes" / "tools" / "uv-0.12.3-darwin-arm64"
    tools.mkdir(parents=True)
    uvx = tools / "uvx"
    uvx.write_text(f"#!{sys.executable}\nprint('synthetic uvx')\n", encoding="utf-8")
    uvx.chmod(0o755)
    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()
    monkeypatch.setenv("PATH", str(empty_path))
    monkeypatch.setenv("PYTHONPATH", "/synthetic/parent/site-packages")
    monkeypatch.setattr(embedded.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        embedded,
        "_DAEMON_START_SNIPPET",
        "import os, subprocess, sys\n"
        "assert 'PYTHONPATH' not in os.environ\n"
        "sys.exit(subprocess.run(['uvx', '--version']).returncode)\n",
    )
    before = dict(os.environ)

    assert embedded._start_daemon_in_clean_child({}, "synthetic-test")
    assert dict(os.environ) == before


@pytest.mark.parametrize("case", ["existing-path", "newest", "non-executable", "absent"])
def test_managed_uvx_selection_preserves_child_environment(monkeypatch, tmp_path, case):
    base_path = tmp_path / "bin"
    base_path.mkdir()
    expected = str(base_path)
    if case != "absent":
        for name, timestamp in [("uv-old", 10), ("uv-new", 20)]:
            tool_dir = tmp_path / ".hermes" / "tools" / name
            tool_dir.mkdir(parents=True)
            uvx = tool_dir / "uvx"
            uvx.write_text("synthetic executable", encoding="utf-8")
            uvx.chmod(0o644 if case == "non-executable" else 0o755)
            os.utime(uvx, (timestamp, timestamp))
        if case == "newest":
            expected = str(tmp_path / ".hermes" / "tools" / "uv-new") + os.pathsep + expected
        if case == "existing-path":
            existing = base_path / "uvx"
            existing.write_text("synthetic executable", encoding="utf-8")
            existing.chmod(0o755)

    monkeypatch.setenv("PATH", str(base_path))
    monkeypatch.setenv("PYTHONPATH", "/synthetic/parent")
    monkeypatch.setattr(embedded.Path, "home", classmethod(lambda cls: tmp_path))
    before = dict(os.environ)
    recorded = {}

    def run(cmd, **kwargs):
        recorded.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(embedded.subprocess, "run", run)
    assert embedded._start_daemon_in_clean_child({}, "synthetic-test")
    assert recorded["env"]["PATH"] == expected
    assert "PYTHONPATH" not in recorded["env"]
    assert dict(os.environ) == before
