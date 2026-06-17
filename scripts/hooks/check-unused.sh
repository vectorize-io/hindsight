#!/bin/bash
# Advisory dead-code scan: surfaces unused functions/methods/classes (Python,
# via vulture) and unused files/exports/dependencies (TypeScript, via knip).
#
# This complements the BLOCKING checks already enforced by ./scripts/hooks/lint.sh
# + the verify-generated-files CI job, where ruff catches unused imports (F401)
# and unused variables (F841). Those tools cannot see whole unused functions,
# orphaned React components, or stale package.json deps — that is what this
# script reports.
#
# This script is ADVISORY (always exits 0): vulture's function/argument
# heuristics produce false positives against FastAPI / SQLAlchemy / Pydantic /
# ABC patterns, so its output is a review aid, not a gate. The CI job
# (check-unused-code) additionally runs `knip --include files,dependencies` as a
# separate BLOCKING step — orphaned files and dead package.json deps are
# unambiguous and fail the build.
#
# Usage: ./scripts/hooks/check-unused.sh

set -u

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

section() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

# --- Python: vulture (dead functions / classes / unreachable code) -----------
# min-confidence 80 keeps noise low; lower it locally (e.g. --min-confidence 60)
# when hunting dead functions, accepting more false positives.
# Format: "<package-dir>:<importable source dir(s)>".
for entry in \
  "hindsight-api-slim:hindsight_api" \
  "hindsight-dev:hindsight_dev benchmarks" \
  "hindsight-embed:hindsight_embed"; do
  pkg="${entry%%:*}"
  srcs="${entry#*:}"
  section "vulture: $pkg"
  (cd "$REPO_ROOT/$pkg" && uvx vulture $srcs --min-confidence 80 2>&1) || true
done

# --- TypeScript: knip (unused files / exports / dependencies) -----------------
section "knip: hindsight-control-plane"
(cd "$REPO_ROOT/hindsight-control-plane" && npx --yes knip@5 --no-progress 2>&1) || true

printf '\n\033[2m(advisory — informational only, does not fail the build)\033[0m\n'
exit 0
