#!/usr/bin/env bash
# Launch the Hindsight MCP server inside the plugin's persistent venv.
# Bootstraps the venv if missing; otherwise just execs.
#
# Unlike the Claude Code plugin's run_mcp.sh, there's no CLAUDE_PLUGIN_ROOT /
# CLAUDE_PLUGIN_DATA equivalent for Devin CLI — scripts/install.py registers
# this script in ~/.config/devin/mcp_config.json with its own absolute path,
# so it resolves everything else (plugin root, data dir) from that.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$(dirname "${SCRIPT_DIR}")"
DATA_DIR="${HINDSIGHT_DEVIN_CLI_DATA_DIR:-${HOME}/.hindsight/devin-cli}"
# Absolutised before anything is derived from it. This script cd's into
# DATA_DIR before the final exec, and POSIX exec resolves a path containing a
# slash against the *current* directory — so a relative override would send
# `exec` looking for the interpreter at DATA_DIR/DATA_DIR/venv/bin/python.
# mkdir first: the directory may not exist yet on a cold start, and `cd` into
# a missing directory would abort the script under `set -e`.
mkdir -p "${DATA_DIR}"
DATA_DIR="$(cd "${DATA_DIR}" && pwd)"
VENV="${DATA_DIR}/venv"
REQ_SRC="${PLUGIN_ROOT}/requirements.txt"
REQ_CACHED="${DATA_DIR}/requirements.txt"

# Resolve the venv interpreter. On Windows-built venvs `bin/python` is
# `python.exe`; bash's `[ -x ]` does not honor PATHEXT, so probe both forms.
# Standard Windows CPython puts the interpreter under `Scripts/` instead.
resolve_py() {
  if [ -x "${VENV}/bin/python" ]; then
    PY="${VENV}/bin/python"
    PIP="${VENV}/bin/pip"
  elif [ -x "${VENV}/bin/python.exe" ]; then
    PY="${VENV}/bin/python.exe"
    PIP="${VENV}/bin/pip.exe"
  elif [ -x "${VENV}/Scripts/python.exe" ]; then
    PY="${VENV}/Scripts/python.exe"
    PIP="${VENV}/Scripts/pip.exe"
  else
    PY=""
    PIP=""
  fi
}

resolve_py
if [ -z "${PY}" ]; then
  if ! python3 -m venv "${VENV}" 2>/dev/null; then
    python -m venv "${VENV}"
  fi
  resolve_py
  if [ -z "${PY}" ]; then
    echo "[Hindsight MCP] venv create failed: no python interpreter at ${VENV}/bin/ or ${VENV}/Scripts/" >&2
    exit 1
  fi
fi

# Re-pip only when the requirements cache is missing, requirements drifted, or
# `mcp` is not importable from the venv.
if [ ! -f "${REQ_CACHED}" ] \
   || ! diff -q "${REQ_SRC}" "${REQ_CACHED}" >/dev/null 2>&1 \
   || ! "${PY}" -c "import mcp" >/dev/null 2>&1; then
  "${PIP}" install --quiet -r "${REQ_SRC}"
  cp "${REQ_SRC}" "${REQ_CACHED}"
fi

# Preserve the session's project directory for bank derivation, then run from
# plugin-owned data so an unreadable project-local .env can't break FastMCP's
# default settings loader.
export HINDSIGHT_MCP_PROJECT_CWD="${PWD}"
cd "${DATA_DIR}"

exec "${PY}" "${PLUGIN_ROOT}/scripts/mcp_server.py"
