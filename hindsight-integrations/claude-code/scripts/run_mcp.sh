#!/usr/bin/env bash
# Launch the Hindsight MCP server inside the plugin's persistent venv.
# Bootstraps the venv if missing; otherwise just execs.
set -e

VENV="${CLAUDE_PLUGIN_DATA}/venv"
REQ_SRC="${CLAUDE_PLUGIN_ROOT}/requirements.txt"
REQ_CACHED="${CLAUDE_PLUGIN_DATA}/requirements.txt"

if [ ! -x "${VENV}/bin/python" ] || ! diff -q "${REQ_SRC}" "${REQ_CACHED}" >/dev/null 2>&1; then
  mkdir -p "${CLAUDE_PLUGIN_DATA}"
  if ! python3 -m venv "${VENV}" 2>/dev/null; then
    python -m venv "${VENV}"
  fi
  "${VENV}/bin/pip" install --quiet -r "${REQ_SRC}"
  cp "${REQ_SRC}" "${REQ_CACHED}"
fi

exec "${VENV}/bin/python" "${CLAUDE_PLUGIN_ROOT}/scripts/mcp_server.py"
