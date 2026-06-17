"""Regression tests for scripts/run_mcp.sh venv interpreter resolution.

`run_mcp.sh` resolves the plugin venv's Python interpreter with a bash
`resolve_py()` helper before either exec-ing it or (if missing) re-creating
the venv. The probe must understand every venv layout the plugin can run on:

  - POSIX:                       ``<venv>/bin/python``
  - msys2 / mingw (git-bash):    ``<venv>/bin/python.exe``  (added in #1564)
  - standard Windows CPython:    ``<venv>/Scripts/python.exe``

The third layout is what the python.org installer, the Windows Store Python,
and ``py -m venv`` all produce — the *common* Windows case. Before the
``Scripts/`` branch existed, ``resolve_py`` returned an empty ``$PY`` on those
venvs, so the launcher fell through to venv re-creation, which then failed
whenever ``python``/``python3`` were not on the spawning process's PATH (issue
#1758, sub-item 3a). These tests invoke the real bash function against a
fabricated venv tree and assert on the interpreter it selects.
"""

import os
import subprocess
import textwrap

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
RUN_MCP_SH = os.path.abspath(os.path.join(SCRIPTS_DIR, "run_mcp.sh"))


def _resolve_py(venv_dir: str) -> str:
    """Run run_mcp.sh's ``resolve_py`` against ``venv_dir`` and return ``$PY``.

    The script's top-level body (which would actually create a venv and exec a
    server) is never run: a tiny driver sources only the ``resolve_py``
    function definition out of the file, sets ``VENV``, calls it, and echoes
    ``$PY``. This keeps the test hermetic and exercises the exact shell logic
    that ships.
    """
    driver = textwrap.dedent(
        """
        set -e
        # Extract just the resolve_py function definition from the script and
        # evaluate it, so none of the script's top-level side effects run.
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


class TestResolvePyVenvLayouts:
    """`resolve_py` must find the interpreter in every supported venv layout."""

    def test_windows_scripts_layout_is_resolved(self, tmp_path):
        """Standard Windows CPython venv: interpreter lives in ``Scripts/``.

        RED before the fix: ``resolve_py`` only probed ``bin/`` and returned an
        empty ``$PY`` for this (common) layout, triggering a doomed venv
        re-creation (issue #1758/3a). GREEN after the ``Scripts/`` branch.
        """
        venv = tmp_path / "venv"
        scripts_python = venv / "Scripts" / "python.exe"
        _make_executable(str(scripts_python))
        # Deliberately no bin/ — this is the genuine Windows layout.
        assert not (venv / "bin").exists()

        resolved = _resolve_py(str(venv))

        assert resolved == str(scripts_python), (
            "resolve_py must select <venv>/Scripts/python.exe for a standard "
            "Windows CPython venv; got: " + repr(resolved)
        )

    def test_posix_bin_layout_is_resolved(self, tmp_path):
        """Regression guard: the POSIX ``bin/python`` path still wins."""
        venv = tmp_path / "venv"
        bin_python = venv / "bin" / "python"
        _make_executable(str(bin_python))

        resolved = _resolve_py(str(venv))

        assert resolved == str(bin_python), (
            "resolve_py must still select <venv>/bin/python on POSIX; got: "
            + repr(resolved)
        )
