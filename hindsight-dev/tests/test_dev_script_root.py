"""Dev entrypoints must run their own checkout, regardless of the caller's cwd."""

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
class RecordedCall:
    command: str
    args: list[str]
    cwd: str
    port: str | None
    dataplane: str | None


@pytest.mark.parametrize("caller", ["checkout", "outside", "other-checkout"])
@pytest.mark.parametrize("entrypoint", ["start-control-plane.sh", "start.sh"])
def test_dev_entrypoint_resolves_its_own_checkout(tmp_path: Path, caller: str, entrypoint: str) -> None:
    root = tmp_path / "hindsight checkout"
    scripts = root / "scripts" / "dev"
    scripts.mkdir(parents=True)
    for name in ("start.sh", "start-api.sh", "start-control-plane.sh"):
        shutil.copy2(SOURCE_ROOT / "scripts" / "dev" / name, scripts / name)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".env").write_text("HINDSIGHT_API_PORT=18888\nHINDSIGHT_CP_PORT=19999\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    other = tmp_path / "other checkout"
    other.mkdir()
    subprocess.run(["git", "init", "-q", str(other)], check=True)
    (other / ".env").write_text("HINDSIGHT_API_PORT=28888\nHINDSIGHT_CP_PORT=29999\n")
    cwd = root if caller == "checkout" else outside if caller == "outside" else other

    # Execute the real entrypoints without starting a server, installing packages,
    # making network requests, or depending on a developer's installed toolchains.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python3").symlink_to(sys.executable)
    calls = tmp_path / "calls.jsonl"
    recorder = """#!/usr/bin/env python3
import json, os, sys, time
from dataclasses import dataclass, asdict
@dataclass
class RecordedCall:
    command: str
    args: list[str]
    cwd: str
    port: str | None
    dataplane: str | None
with open(os.environ['DEV_SCRIPT_CALLS'], 'a') as output:
    call = RecordedCall(os.path.basename(sys.argv[0]), sys.argv[1:], os.getcwd(),
                        os.environ.get('PORT'), os.environ.get('HINDSIGHT_CP_DATAPLANE_API_URL'))
    output.write(json.dumps(asdict(call)) + '\\n')
if os.path.basename(sys.argv[0]) == 'npm' and 'dev' in sys.argv:
    open(os.environ['DEV_SCRIPT_DONE'], 'w').close()
if os.path.basename(sys.argv[0]) == 'uv':
    deadline = time.monotonic() + 5
    while not os.path.exists(os.environ['DEV_SCRIPT_DONE']) and time.monotonic() < deadline:
        time.sleep(0.02)
if os.path.basename(sys.argv[0]) == 'sleep':
    time.sleep(0.05)
sys.exit(9 if os.path.basename(sys.argv[0]) == 'uv' else 0)
"""
    for command in ("npm", "uv", "curl", "sleep"):
        path = bin_dir / command
        path.write_text(recorder)
        path.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("HINDSIGHT_") and k != "PORT"}
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["DEV_SCRIPT_CALLS"] = str(calls)
    env["DEV_SCRIPT_DONE"] = str(tmp_path / "dev-done")
    result = subprocess.run(
        ["bash", str(scripts / entrypoint)], cwd=cwd, env=env, capture_output=True, text=True, timeout=10
    )
    records = [RecordedCall(**json.loads(line)) for line in calls.read_text().splitlines()] if calls.exists() else []
    npm_calls = [call for call in records if call.command == "npm"]
    assert len(npm_calls) == 2, result.stdout + result.stderr
    assert all(call.cwd == str(root) for call in npm_calls)
    dev_call = next(call for call in npm_calls if "dev" in call.args)
    assert dev_call.port == "19999"
    assert dev_call.dataplane == ("http://localhost:18888" if entrypoint == "start.sh" else "http://localhost:8888")
    if entrypoint == "start.sh":
        assert result.returncode == 1  # The stubbed API exits; the real cleanup path is retained.
        api_call = next(call for call in records if call.command == "uv")
        assert api_call.cwd == str(root)
        assert api_call.args == ["run", "hindsight-api", "--port", "18888"]
    else:
        assert result.returncode == 0, result.stdout + result.stderr
