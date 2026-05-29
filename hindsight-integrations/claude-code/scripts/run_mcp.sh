#!/usr/bin/env bash
# Launch the Hindsight MCP server inside the plugin's persistent venv.
# Bootstraps the venv if missing; otherwise just execs.
set -e

VENV="${CLAUDE_PLUGIN_DATA}/venv"
REQ_SRC="${CLAUDE_PLUGIN_ROOT}/requirements.txt"
REQ_CACHED="${CLAUDE_PLUGIN_DATA}/requirements.txt"

# Resolve the venv interpreter. On Windows-built venvs `bin/python` is
# `python.exe`; bash's `[ -x ]` does not honor PATHEXT, so probe both forms.
resolve_py() {
  if [ -x "${VENV}/bin/python" ]; then
    PY="${VENV}/bin/python"
    PIP="${VENV}/bin/pip"
  elif [ -x "${VENV}/bin/python.exe" ]; then
    PY="${VENV}/bin/python.exe"
    PIP="${VENV}/bin/pip.exe"
  else
    PY=""
    PIP=""
  fi
}

resolve_py
if [ -z "${PY}" ]; then
  mkdir -p "${CLAUDE_PLUGIN_DATA}"
  if ! python3 -m venv "${VENV}" 2>/dev/null; then
    python -m venv "${VENV}"
  fi
  resolve_py
  if [ -z "${PY}" ]; then
    echo "[Hindsight MCP] venv create failed: no python interpreter at ${VENV}/bin/" >&2
    exit 1
  fi
fi

# Re-pip only when the requirements cache is missing, requirements drifted, or
# `mcp` is not importable from the venv. Splitting this from venv creation
# keeps warm starts cheap and avoids re-running pip over a venv that's already
# in use (which fails with ERROR_SHARING_VIOLATION on Windows).
if [ ! -f "${REQ_CACHED}" ] \
   || ! diff -q "${REQ_SRC}" "${REQ_CACHED}" >/dev/null 2>&1 \
   || ! "${PY}" -c "import mcp" >/dev/null 2>&1; then
  "${PIP}" install --quiet -r "${REQ_SRC}"
  cp "${REQ_SRC}" "${REQ_CACHED}"
fi

exec "${PY}" "${CLAUDE_PLUGIN_ROOT}/scripts/mcp_server.py"
