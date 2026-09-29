#!/bin/bash
# Document-evolution benchmark: does a living document survive being edited?
#
# Runs against an already-running Hindsight server, so the same harness can
# drive a server built from any revision — that is how two builds get compared.
# See hindsight-dev/benchmarks/document_evolution/README.md.
#
#   ./scripts/benchmarks/run-document-evolution.sh run --api-url http://localhost:8888 --build main
#   ./scripts/benchmarks/run-document-evolution.sh compare results/main.json results/branch.json
set -e

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR/hindsight-dev"
exec uv run python -m benchmarks.document_evolution "$@"
