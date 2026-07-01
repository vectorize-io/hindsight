#!/bin/bash
# verify.sh - Wrapper for oma verify
# Usage: ./verify.sh <agent-type> [workspace-path]

AGENT_TYPE="${1:-}"
WORKSPACE="${2:-.}"

exec oma verify "$AGENT_TYPE" --workspace "$WORKSPACE"
