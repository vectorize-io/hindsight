#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

fail() {
    echo "FAIL: $1" >&2
    exit 1
}

write_manifest() {
    local path=$1 version=$2
    mkdir -p "$(dirname "$path")"
    printf '{\n  "version": "%s"\n}\n' "$version" > "$path"
}

# Exercise the real integration release script in an isolated repository. The
# check path must not create tags, push, or require changelog credentials.
INTEGRATION_REPO="$TMP_DIR/integration"
mkdir -p "$INTEGRATION_REPO/scripts" "$INTEGRATION_REPO/hindsight-integrations/claude-code"
cp "$ROOT/scripts/release-integration.sh" "$INTEGRATION_REPO/scripts/release-integration.sh"
write_manifest "$INTEGRATION_REPO/hindsight-integrations/claude-code/.claude-plugin/plugin.json" "0.7.5"
write_manifest "$INTEGRATION_REPO/.claude-plugin/marketplace.json" "0.7.5"
printf 'initial\n' > "$INTEGRATION_REPO/hindsight-integrations/claude-code/integration.txt"

git -C "$INTEGRATION_REPO" init -q -b main
git -C "$INTEGRATION_REPO" config user.name "Release Test"
git -C "$INTEGRATION_REPO" config user.email "release-test@example.com"
git -C "$INTEGRATION_REPO" add .
git -C "$INTEGRATION_REPO" commit -q -m "initial integration"
git -C "$INTEGRATION_REPO" tag "integrations/claude-code/v0.7.5"
printf 'changed\n' >> "$INTEGRATION_REPO/hindsight-integrations/claude-code/integration.txt"
git -C "$INTEGRATION_REPO" add .
git -C "$INTEGRATION_REPO" commit -q -m "change integration"

set +e
STALE_OUTPUT=$(cd "$INTEGRATION_REPO" && ./scripts/release-integration.sh --check claude-code 2>&1)
STALE_STATUS=$?
set -e
[ "$STALE_STATUS" -ne 0 ] || fail "stale Claude Code manifest passed the release check"
case "$STALE_OUTPUT" in
    *"unreleased changes"*) ;;
    *) fail "stale release check did not explain the unreleased changes: $STALE_OUTPUT" ;;
esac

write_manifest "$INTEGRATION_REPO/hindsight-integrations/claude-code/.claude-plugin/plugin.json" "0.7.6"
write_manifest "$INTEGRATION_REPO/.claude-plugin/marketplace.json" "0.7.6"
git -C "$INTEGRATION_REPO" add .
git -C "$INTEGRATION_REPO" commit -q -m "release claude-code 0.7.6"
git -C "$INTEGRATION_REPO" tag "integrations/claude-code/v0.7.6"
(cd "$INTEGRATION_REPO" && ./scripts/release-integration.sh --check claude-code)

# Nested plugin entries may carry their own versions. The guard must read the
# top-level marketplace version regardless of JSON key order.
cat > "$INTEGRATION_REPO/.claude-plugin/marketplace.json" <<'JSON'
{
  "plugins": [{"name": "hindsight-memory", "version": "9.9.9"}],
  "version": "0.7.6"
}
JSON
(cd "$INTEGRATION_REPO" && ./scripts/release-integration.sh --check claude-code)

write_manifest "$INTEGRATION_REPO/.claude-plugin/marketplace.json" "0.8.4"
set +e
MISMATCH_OUTPUT=$(cd "$INTEGRATION_REPO" && ./scripts/release-integration.sh --check claude-code 2>&1)
MISMATCH_STATUS=$?
set -e
[ "$MISMATCH_STATUS" -ne 0 ] || fail "mismatched Claude Code manifests passed the release check"
case "$MISMATCH_OUTPUT" in
    *"must match"*) ;;
    *) fail "manifest mismatch did not produce an actionable error: $MISMATCH_OUTPUT" ;;
esac

set +e
EXTRA_ARG_OUTPUT=$(cd "$INTEGRATION_REPO" && ./scripts/release-integration.sh --check claude-code 0.9.9 2>&1)
EXTRA_ARG_STATUS=$?
set -e
[ "$EXTRA_ARG_STATUS" -ne 0 ] || fail "--check silently accepted an unexpected version argument"
case "$EXTRA_ARG_OUTPUT" in
    *"Usage:"*) ;;
    *) fail "extra --check argument did not show usage: $EXTRA_ARG_OUTPUT" ;;
esac

# The core release path must invoke the independent integration guard before it
# mutates files or reaches any tag/push/changelog work.
CORE_REPO="$TMP_DIR/core"
mkdir -p "$CORE_REPO/scripts"
cp "$ROOT/scripts/release.sh" "$CORE_REPO/scripts/release.sh"
printf '#!/bin/bash\nprintf "%%s\\n" "$*" > "$CHECK_LOG"\nexit 42\n' > "$CORE_REPO/scripts/release-integration.sh"
chmod +x "$CORE_REPO/scripts/release-integration.sh"
git -C "$CORE_REPO" init -q -b main
git -C "$CORE_REPO" config user.name "Release Test"
git -C "$CORE_REPO" config user.email "release-test@example.com"
printf 'fixture\n' > "$CORE_REPO/README.md"
git -C "$CORE_REPO" add .
git -C "$CORE_REPO" commit -q -m "core fixture"

CHECK_LOG="$TMP_DIR/check-args"
set +e
CORE_OUTPUT=$(cd "$CORE_REPO" && CHECK_LOG="$CHECK_LOG" ./scripts/release.sh 9.9.9 </dev/null 2>&1)
CORE_STATUS=$?
set -e
[ "$CORE_STATUS" -eq 42 ] || fail "core release bypassed the integration guard (status $CORE_STATUS): $CORE_OUTPUT"
[ "$(cat "$CHECK_LOG")" = "--check claude-code" ] || fail "core release called the guard with unexpected arguments"
[ ! -e "$CORE_REPO/.git/refs/tags/v9.9.9" ] || fail "core release created a tag after the guard failed"

# Core releases must refresh the generated docs skill after regenerating the
# OpenAPI spec and clients so committed copies cannot drift between releases.
grep -q 'generate-openapi.sh.*generate-clients.sh.*generate-docs-skill.sh' "$ROOT/scripts/release.sh" || \
    fail "core release does not regenerate the docs skill after OpenAPI and clients"

PLUGIN_VERSION=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["version"])' \
    "$ROOT/hindsight-integrations/claude-code/.claude-plugin/plugin.json")
MARKETPLACE_VERSION=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["version"])' \
    "$ROOT/.claude-plugin/marketplace.json")
[ "$PLUGIN_VERSION" = "0.7.6" ] || fail "plugin manifest version is $PLUGIN_VERSION, expected 0.7.6"
[ "$MARKETPLACE_VERSION" = "0.7.6" ] || fail "marketplace version is $MARKETPLACE_VERSION, expected 0.7.6"

echo "PASS: Claude Code release guard regression tests"
