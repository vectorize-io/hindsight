#!/usr/bin/env bash
#
# Self-test for smoke-test.sh's assert_doctor_clean.
#
# The doctor assertion is the only thing standing between "the plugin loads but
# silently never recalls or retains" and a green smoke run, and it already
# missed that case once: it matched a word list ('fail', 'error', 'not loaded'),
# which does not appear in openclaw 2026.9.x's "typed hook ... blocked because
# ..." diagnostic. The fixtures below are real captured `openclaw plugins
# doctor` output, so a regression in the assertion fails here rather than in
# production six weeks later.
#
# Usage: ./scripts/doctor-check-selftest.sh [path-to-smoke-test.sh]
set -uo pipefail
GREEN=''; RED=''; YELLOW=''; NC=''
log() { :; }
warn() { :; }
fail() { echo "FAILED: $*"; exit 42; }
# shellcheck disable=SC1090
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE="${1:-$SCRIPT_DIR/smoke-test.sh}"
eval "$(sed -n '/^assert_doctor_clean()/,/^}/p' "$SMOKE")"

BLOCKED='Diagnostics:
- memory-core: memory plugin not selected for the memory slot; skipping its indexing runtime and recall registration (consolidation lifecycle preserved)
- hindsight-openclaw: typed hook "before_prompt_build" blocked because non-bundled plugins must set plugins.entries.hindsight-openclaw.hooks.allowConversationAccess=true
- hindsight-openclaw: typed hook "agent_end" blocked because non-bundled plugins must set plugins.entries.hindsight-openclaw.hooks.allowConversationAccess=true'

HEALTHY='Diagnostics:
- memory-core: memory plugin not selected for the memory slot; skipping its indexing runtime and recall registration (consolidation lifecycle preserved)'

CLEAN='No plugin issues detected.'

UNRELATED='Diagnostics:
- ollama: provider registered twice'

check() { # name expected_rc output rc
  ( assert_doctor_clean "$3" "$4" "test" >/dev/null 2>&1 ); local got=$?
  if [[ $got -eq $2 ]]; then echo "PASS  $1"; else echo "FAIL  $1 (expected exit $2, got $got)"; FAILED=1; fi
}
FAILED=0
check "blocked hooks (2026.9.x) must FAIL"        42 "$BLOCKED"   1
check "memory-slot notice only must PASS"          0 "$HEALTHY"   1
check "clean doctor must PASS"                     0 "$CLEAN"     0
check "unrelated diag, exit 0, must PASS"          0 "$UNRELATED" 0
check "unrelated diag, exit 1, must FAIL"         42 "$UNRELATED" 1
if [[ $FAILED -eq 0 ]]; then echo "all assert_doctor_clean cases passed"; fi
exit $FAILED
