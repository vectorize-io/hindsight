"""The tool version edit must work with the host sed without releasing anything."""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class RecordedAction:
    command: str
    args: list[str]


@pytest.mark.parametrize("manifest", ["package.json", "pyproject.toml"])
def test_release_tool_updates_version_with_host_sed(tmp_path: Path, manifest: str) -> None:
    root = tmp_path / "checkout"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(SOURCE_ROOT / "scripts" / "release-tool.sh", scripts / "release-tool.sh")
    tool = root / "hindsight-tools" / "hindsight-agent-sdk"
    tool.mkdir(parents=True)
    old = '{"name": "fixture", "version": "0.1.0"}\n' if manifest == "package.json" else 'version = "0.1.0"\n'
    (tool / manifest).write_text(old)
    calls = tmp_path / "calls.jsonl"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python3").symlink_to(sys.executable)
    # Run the real release script and host sed, but record every Git/build action
    # instead of creating a release commit, tag, remote push or build artifact.
    recorder = """#!/usr/bin/env python3
import json, os, sys
from dataclasses import dataclass, asdict
@dataclass
class RecordedAction:
    command: str
    args: list[str]
with open(os.environ['RELEASE_TEST_CALLS'], 'a') as output:
    output.write(json.dumps(asdict(RecordedAction(os.path.basename(sys.argv[0]), sys.argv[1:]))) + '\\n')
if sys.argv[1:] == ['branch', '--show-current']:
    print('main')
"""
    for command in ("git", "npm"):
        path = bin_dir / command
        path.write_text(recorder)
        path.chmod(0o755)
    env = os.environ | {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "RELEASE_TEST_CALLS": str(calls)}
    result = subprocess.run(
        ["bash", str(scripts / "release-tool.sh"), "hindsight-agent-sdk", "0.2.0"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tool / manifest).read_text() == old.replace("0.1.0", "0.2.0")
    assert not list(tool.glob("*.bak"))
    records = [RecordedAction(**json.loads(line)) for line in calls.read_text().splitlines()]
    git_calls = [call.args for call in records if call.command == "git"]
    assert git_calls == [
        ["branch", "--show-current"],
        ["status", "--porcelain"],
        ["add", "hindsight-tools/hindsight-agent-sdk/"],
        ["commit", "-m", "release(hindsight-agent-sdk): v0.2.0"],
        ["tag", "tools/hindsight-agent-sdk/v0.2.0"],
        ["push", "origin", "main", "tools/hindsight-agent-sdk/v0.2.0"],
    ]
    assert [call.args for call in records if call.command == "npm"] == (
        [["run", "build"]] if manifest == "package.json" else []
    )
