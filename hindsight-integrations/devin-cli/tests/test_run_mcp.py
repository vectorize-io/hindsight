"""Regression tests for scripts/run_mcp.sh, ported from the Claude Code plugin.

`run_mcp.sh` resolves the plugin venv's Python interpreter with a bash
`resolve_py()` helper before either exec-ing it or (if missing) re-creating the
venv. The probe must understand every venv layout the plugin can run on:

  - POSIX:                       ``<venv>/bin/python``
  - msys2 / mingw (git-bash):    ``<venv>/bin/python.exe``
  - standard Windows CPython:    ``<venv>/Scripts/python.exe``

The third is what the python.org installer, the Windows Store Python, and
``py -m venv`` all produce. On the Claude Code side, a missing ``Scripts/``
branch returned an empty ``$PY``, so the launcher fell through to venv
re-creation, which then failed whenever ``python``/``python3`` were not on the
spawning process's PATH.

These invoke the real bash function against a fabricated venv tree rather than
grepping the script, so they fail if the logic breaks and not merely if it is
reworded.
"""

import os
import stat
import subprocess
import textwrap

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
RUN_MCP_SH = os.path.abspath(os.path.join(SCRIPTS_DIR, "run_mcp.sh"))


def _resolve_py(venv_dir: str) -> str:
    """Run run_mcp.sh's ``resolve_py`` against ``venv_dir`` and return ``$PY``.

    A tiny driver sources only the ``resolve_py`` function definition out of
    the file, sets ``VENV``, calls it, and echoes ``$PY`` — so none of the
    script's top-level side effects (venv creation, pip install, exec) run.
    """
    driver = textwrap.dedent(
        """
        set -e
        func="$(sed -n '/^resolve_py() {/,/^}/p' "$1")"
        eval "$func"
        VENV="$2"
        resolve_py
        printf '%s' "$PY"
        """
    )
    result = subprocess.run(
        ["bash", "-c", driver, "bash", RUN_MCP_SH, venv_dir],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _make_executable(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("#!/usr/bin/env python\n")
    os.chmod(path, 0o755)


def _run_tail_with_fake_exec(data_dir: str) -> str:
    """Run the launcher tail with a fake exec; return project cwd and server cwd."""
    driver = textwrap.dedent(
        """
        set -e
        export HINDSIGHT_DEVIN_CLI_DATA_DIR="$1"
        DATA_DIR="$1"
        # Sourcing only the tail skips the head's mkdir/absolutise step, so the
        # directory it cd's into has to be staged here.
        mkdir -p "${DATA_DIR}"
        PLUGIN_ROOT="$(dirname "$(dirname "$2")")"
        PY=python
        exec() { printf '%s\n%s' "$HINDSIGHT_MCP_PROJECT_CWD" "$(pwd)"; }
        sed -n '/^export HINDSIGHT_MCP_PROJECT_CWD=/,$p' "$2" | source /dev/stdin
        """
    )
    result = subprocess.run(
        ["bash", "-c", driver, "bash", data_dir, RUN_MCP_SH],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


class TestResolvePyVenvLayouts:
    """`resolve_py` must find the interpreter in every supported venv layout."""

    def test_windows_scripts_layout_is_resolved(self, tmp_path):
        venv = tmp_path / "venv"
        scripts_python = venv / "Scripts" / "python.exe"
        _make_executable(str(scripts_python))
        assert not (venv / "bin").exists()

        assert _resolve_py(str(venv)) == str(scripts_python), (
            "resolve_py must select <venv>/Scripts/python.exe for a standard Windows CPython venv"
        )

    def test_posix_bin_layout_is_resolved(self, tmp_path):
        venv = tmp_path / "venv"
        bin_python = venv / "bin" / "python"
        _make_executable(str(bin_python))

        assert _resolve_py(str(venv)) == str(bin_python)

    def test_msys_bin_exe_layout_is_resolved(self, tmp_path):
        venv = tmp_path / "venv"
        bin_python_exe = venv / "bin" / "python.exe"
        _make_executable(str(bin_python_exe))

        assert _resolve_py(str(venv)) == str(bin_python_exe)

    def test_empty_venv_yields_empty_py_so_the_caller_recreates_it(self, tmp_path):
        venv = tmp_path / "venv"
        venv.mkdir()

        assert _resolve_py(str(venv)) == ""


class TestMcpServerWorkingDirectory:
    """The launcher must not run the server from the session's project cwd."""

    def test_exec_runs_from_data_dir_but_preserves_project_cwd(self, tmp_path):
        """FastMCP probes `.env` in cwd, so exec from the plugin-owned data dir.

        The project cwd still has to survive, because `derive_bank_id` uses it
        — hence the HINDSIGHT_MCP_PROJECT_CWD export before the `cd`.
        """
        project = tmp_path / "project"
        project.mkdir()
        env_file = project / ".env"
        env_file.write_text("SECRET=value\n")
        env_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
        data_dir = tmp_path / "plugin-data"

        previous = os.getcwd()
        try:
            os.chdir(project)
            project_cwd, server_cwd = _run_tail_with_fake_exec(str(data_dir)).splitlines()
        finally:
            os.chdir(previous)

        assert project_cwd == str(project)
        assert server_cwd == str(data_dir)


def _run_full_with_fake_exec(cwd: str, data_dir_value: str) -> str:
    """Run the whole launcher with a fake exec; return "$PY" and the server cwd.

    Unlike `_run_tail_with_fake_exec` this sources the entire script, so the
    DATA_DIR normalisation at the top actually runs.
    """
    driver = textwrap.dedent(
        """
        set -e
        export HINDSIGHT_DEVIN_CLI_DATA_DIR="$1"
        exec() { printf '%s\n%s' "$PY" "$(pwd)"; }
        source "$2"
        """
    )
    result = subprocess.run(
        ["bash", "-c", driver, "bash", data_dir_value, RUN_MCP_SH],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=True,
    )
    return result.stdout.strip()


class TestRelativeDataDirOverride:
    """`exec` resolves a path containing a slash against the *current* directory.

    The launcher cd's into DATA_DIR before exec-ing the interpreter, so a
    relative HINDSIGHT_DEVIN_CLI_DATA_DIR used to produce a doubled path
    (`reldata/reldata/venv/bin/python`) and the MCP server never started.
    """

    def _fake_venv(self, data_dir):
        bin_dir = os.path.join(data_dir, "venv", "bin")
        os.makedirs(bin_dir, exist_ok=True)
        for name in ("python", "pip"):
            path = os.path.join(bin_dir, name)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("#!/bin/sh\nexit 0\n")
            os.chmod(path, 0o755)
        # Matching the cached requirements skips the pip branch entirely.
        req_src = os.path.abspath(os.path.join(SCRIPTS_DIR, "..", "requirements.txt"))
        with open(req_src, encoding="utf-8") as fh:
            contents = fh.read()
        with open(os.path.join(data_dir, "requirements.txt"), "w", encoding="utf-8") as fh:
            fh.write(contents)

    def test_a_relative_data_dir_still_resolves_the_interpreter(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        self._fake_venv(str(work / "reldata"))

        py, server_cwd = _run_full_with_fake_exec(str(work), "reldata").splitlines()

        assert os.path.isabs(py), f"interpreter path must be absolute after the cd, got {py!r}"
        assert os.path.exists(py), f"{py!r} does not exist — exec would fail"
        assert os.path.realpath(server_cwd) == os.path.realpath(str(work / "reldata"))

    def test_an_absolute_data_dir_is_unaffected(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        data_dir = tmp_path / "absdata"
        self._fake_venv(str(data_dir))

        py, server_cwd = _run_full_with_fake_exec(str(work), str(data_dir)).splitlines()

        assert os.path.exists(py)
        assert os.path.realpath(server_cwd) == os.path.realpath(str(data_dir))

    def test_a_missing_data_dir_is_created_rather_than_aborting(self, tmp_path):
        """`cd` into a nonexistent directory under `set -e` would kill the script."""
        work = tmp_path / "work"
        work.mkdir()

        result = subprocess.run(
            [
                "bash",
                "-c",
                'set -e\nexport HINDSIGHT_DEVIN_CLI_DATA_DIR="$1"\nsource "$2"',
                "bash",
                "fresh",
                RUN_MCP_SH,
            ],
            capture_output=True,
            text=True,
            cwd=str(work),
        )

        assert (work / "fresh").is_dir(), f"data dir was not created: {result.stderr}"
