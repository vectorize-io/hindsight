"""Exercise daemon-child descriptor redirection in disposable Python processes."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(os.name != "posix", reason="Windows daemonize does not redirect stdio")
@pytest.mark.parametrize("stdin_state", ["normal", "none", "closed_fd", "closed_stream", "none_closed_fd"])
def test_daemon_child_restores_stdin_and_redirects_output(tmp_path: Path, stdin_state: str) -> None:
    log_path = tmp_path / "logs" / "daemon.log"
    code = """
import os
import sys
from pathlib import Path
from hindsight_api import daemon

# Isolate log-path configuration, not descriptor operations or daemon dispatch.
daemon.daemon_log_path = lambda: Path(sys.argv[2])
state = sys.argv[1]
if state == "closed_stream":
    sys.stdin.close()
if state in ("closed_fd", "none_closed_fd"):
    os.close(0)
if state in ("none", "none_closed_fd"):
    sys.stdin = None

daemon.daemonize()
assert sys.stdin.read() == ""
assert os.read(0, 1) == b""
assert os.get_inheritable(0)
print("python stdout", flush=True)
print("python stderr", file=sys.stderr, flush=True)
os.write(1, b"fd stdout\\n")
os.write(2, b"fd stderr\\n")
"""
    # Do not load the caller's credentials or .env: config and logs belong to this
    # disposable child. Import the checkout, not a separately installed API build.
    env = {
        "PATH": os.defpath,
        "HOME": str(tmp_path),
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
        "HINDSIGHT_API_DAEMON_LOG": str(log_path),
        "_HINDSIGHT_DAEMON_CHILD": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", code, stdin_state, str(log_path)],
        env=env,
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == result.stderr == ""
    assert log_path.read_text() == "python stdout\npython stderr\nfd stdout\nfd stderr\n"
