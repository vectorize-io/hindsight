"""Concurrent `profile create --merge` must never drop keys already in the profile (#6485).

Every coding-agent session start runs that merge. It used to truncate the .env and
rewrite it in place, unlocked, so a session reading it mid-rewrite saw it empty and
wrote back a file without the user's DATABASE_URL: the next daemon start then fell
back to a new, empty pg0 and silently stored memories there.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

DB_LINE = "HINDSIGHT_EMBED_API_DATABASE_URL=postgresql://u:p@127.0.0.1:5433/real"

# Many merges per process: one process start buys many interleavings with the others.
_MERGE_LOOP = """
import sys
from hindsight_embed import cli
for i in range(int(sys.argv[1])):
    sys.argv = ["hindsight-embed", "profile", "create", "coding-agent", "--merge", "--port", "9077",
                "--env", f"HINDSIGHT_API_LLM_MODEL=m{i}"]
    try:
        cli.main()
    except SystemExit as e:
        if e.code:
            raise
"""


def _run_cli(home: Path, *args: str, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "HOME": str(home), "USERPROFILE": str(home), **(env_extra or {})}
    return subprocess.run(
        [sys.executable, "-m", "hindsight_embed.cli", *args], env=env, capture_output=True, text=True, timeout=60
    )


def test_concurrent_merges_keep_the_database_url(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    created = _run_cli(home, "profile", "create", "coding-agent", "--port", "9077", "--env", DB_LINE)
    assert created.returncode == 0, created.stderr

    env = {**os.environ, "HOME": str(home), "USERPROFILE": str(home)}
    procs = [
        subprocess.Popen([sys.executable, "-c", _MERGE_LOOP, "25"], env=env, stdout=subprocess.DEVNULL)
        for _ in range(6)
    ]
    assert all(p.wait(timeout=120) == 0 for p in procs)

    profile = (home / ".hindsight" / "profiles" / "coding-agent.env").read_text(encoding="utf-8")
    assert DB_LINE in profile.splitlines()


def test_merge_waits_on_the_profile_lock_and_reports_a_timeout(tmp_path: Path) -> None:
    from hindsight_embed.profile_manager import lock_file, unlock_file

    home = tmp_path / "home"
    home.mkdir()
    assert _run_cli(home, "profile", "create", "coding-agent", "--port", "9077").returncode == 0

    lock_path = home / ".hindsight" / "profiles" / "coding-agent.env.lock"
    with open(lock_path, "w") as held:
        lock_file(held)
        try:
            result = _run_cli(
                home,
                "profile",
                "create",
                "coding-agent",
                "--merge",
                "--port",
                "9077",
                env_extra={"HINDSIGHT_EMBED_LOCK_TIMEOUT": "0.5"},
            )
        finally:
            unlock_file(held)

    assert result.returncode == 1
    assert "Timed out" in result.stderr


def test_delete_removes_the_merge_lock(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    assert _run_cli(home, "profile", "create", "coding-agent", "--port", "9077").returncode == 0
    lock_path = home / ".hindsight" / "profiles" / "coding-agent.env.lock"
    assert lock_path.exists()

    deleted = _run_cli(home, "profile", "delete", "coding-agent")

    assert deleted.returncode == 0, deleted.stderr
    assert not lock_path.exists()


def test_a_refused_swap_keeps_the_old_profile_and_no_temp_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows refuses the swap while another process has the .env open: nothing may be lost."""
    from hindsight_embed import profile_manager

    monkeypatch.setattr(profile_manager.Path, "home", lambda: tmp_path)
    pm = profile_manager.ProfileManager()
    pm.create_profile("coding-agent", 9077, {"HINDSIGHT_EMBED_API_DATABASE_URL": "postgresql://real"})
    env_path = pm._get_profiles_dir() / "coding-agent.env"
    before = env_path.read_text(encoding="utf-8")

    def refuse(self: Path, target: Path) -> Path:
        raise PermissionError("in use")

    monkeypatch.setattr(profile_manager.Path, "replace", refuse)
    with pytest.raises(PermissionError):
        pm.create_profile("coding-agent", 9077, {})

    assert env_path.read_text(encoding="utf-8") == before
    assert not list(env_path.parent.glob("*.tmp"))
