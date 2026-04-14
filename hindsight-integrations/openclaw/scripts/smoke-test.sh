#!/usr/bin/env bash
#
# End-to-end smoke test for the Hindsight OpenClaw plugin.
#
# What it verifies:
#   1. The plugin tarball installs cleanly via `openclaw plugins install`
#      WITHOUT --dangerously-force-unsafe-install (install scanner reports
#      zero findings).
#   2. Workspace deps (@vectorize-io/hindsight-all, hindsight-client) resolve
#      from the npm registry into the extracted extension's node_modules.
#   3. The non-interactive `hindsight-openclaw-setup` wizard writes a valid
#      openclaw.json plugin config for each of the three modes.
#   4. `openclaw config validate` + `openclaw plugins doctor` pass after each
#      setup run.
#   5. The wizard rejects invalid flag combinations with a non-zero exit code.
#
# What it does NOT do:
#   - Start an openclaw gateway or run agent turns. Those are covered by the
#     existing integration tests (`npm run test:integration`) which exercise
#     the plugin's hook handlers directly against a real Hindsight API via a
#     mock MoltbotPluginAPI. This script targets the install / config path
#     that integration tests can't cover (because they bypass the CLI).
#
# Safety:
#   - Backs up ~/.openclaw/openclaw.json before any mutation and restores it
#     on exit (success or failure).
#   - Removes any pre-existing ~/.openclaw/extensions/hindsight-openclaw dir
#     at start so it runs from a clean slate.
#   - Intended for CI and local dev. In CI the backup/restore is effectively
#     a no-op because there's no prior state.
#
# Usage:
#   ./scripts/smoke-test.sh                # packs a fresh tarball
#   ./scripts/smoke-test.sh <tarball-path> # uses an existing tarball
#
# Requirements: openclaw CLI on PATH, node, npm.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OPENCLAW_HOME="${OPENCLAW_HOME:-$HOME/.openclaw}"
CONFIG_PATH="$OPENCLAW_HOME/openclaw.json"
EXT_DIR="$OPENCLAW_HOME/extensions/hindsight-openclaw"
HINDSIGHT_API_URL="${HINDSIGHT_API_URL:-http://127.0.0.1:7777}"
CONFIG_BACKUP=""
EXT_BACKUP=""
TARBALL=""

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { printf "${GREEN}[smoke-test]${NC} %s\n" "$*"; }
warn() { printf "${YELLOW}[smoke-test]${NC} %s\n" "$*" >&2; }
fail() { printf "${RED}[smoke-test FAIL]${NC} %s\n" "$*" >&2; exit 1; }

require() {
  command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

cleanup() {
  local rc=$?
  log "cleaning up"
  # Always uninstall / remove the smoke-test install first so we start from a
  # known-clean state before attempting restore.
  yes 2>/dev/null | openclaw plugins uninstall hindsight-openclaw >/dev/null 2>&1 || true
  rm -rf "$EXT_DIR"

  # Restore the user's openclaw.json if we backed it up.
  if [[ -n "$CONFIG_BACKUP" && -f "$CONFIG_BACKUP" ]]; then
    mv "$CONFIG_BACKUP" "$CONFIG_PATH"
    log "restored $CONFIG_PATH from backup"
  fi
  # Restore the extension dir if we backed it up — keeps config and files
  # consistent on dev machines that had the plugin installed pre-run.
  if [[ -n "$EXT_BACKUP" && -d "$EXT_BACKUP" ]]; then
    mv "$EXT_BACKUP" "$EXT_DIR"
    log "restored $EXT_DIR from backup"
  fi

  # Clean up the tarball only if we packed it ourselves.
  if [[ -n "$TARBALL" && "$TARBALL" == "$PLUGIN_DIR"/vectorize-io-hindsight-openclaw-*.tgz ]]; then
    rm -f "$TARBALL"
  fi
  exit "$rc"
}

run_setup_mode() {
  local label="$1"
  shift
  log "running setup → $label"
  if ! node "$EXT_DIR/dist/setup.js" --config-path "$CONFIG_PATH" "$@" >/dev/null; then
    fail "hindsight-openclaw-setup --mode failed for: $label"
  fi
  if ! openclaw config validate >/dev/null 2>&1; then
    openclaw config validate >&2 || true
    fail "openclaw config validate failed after: $label"
  fi
  # `openclaw plugins doctor` can print diagnostics for UNRELATED bundled
  # plugins (e.g. ollama double-registration in clean CI envs). Only fail if
  # doctor surfaces something that specifically names hindsight, or if the
  # command itself exits non-zero.
  local doctor_out
  if ! doctor_out="$(openclaw plugins doctor 2>&1)"; then
    printf '%s\n' "$doctor_out" >&2
    fail "openclaw plugins doctor exited non-zero after: $label"
  fi
  if printf '%s' "$doctor_out" | grep -iE 'hindsight.*(fail|error|not loaded)|(fail|error).*hindsight' >/dev/null; then
    printf '%s\n' "$doctor_out" >&2
    fail "plugins doctor reported hindsight-specific issues after: $label"
  fi
  log "  ✓ $label → config valid + doctor clean"
}

get_config_value() {
  openclaw config get "$1" 2>/dev/null | tail -1
}

# Read a value directly from the raw openclaw.json. `openclaw config get`
# redacts sensitive fields (renders them as "__OPENCLAW_REDACTED__"), so we
# can't use it to assert the actual token/key value stored inline. This
# helper bypasses the redaction for smoke-test assertions.
get_raw_config_value() {
  python3 -c "
import json, sys
path = sys.argv[1]
dotted = sys.argv[2]
with open(path) as f:
    node = json.load(f)
for part in dotted.split('.'):
    if isinstance(node, dict) and part in node:
        node = node[part]
    else:
        sys.exit(0)
if isinstance(node, (dict, list)):
    print(json.dumps(node))
else:
    print(node)
" "$CONFIG_PATH" "$1"
}

main() {
  require openclaw
  require node
  require npm

  trap cleanup EXIT

  # Back up any existing openclaw.json + extension dir (no-op in CI).
  if [[ -f "$CONFIG_PATH" ]]; then
    CONFIG_BACKUP="$CONFIG_PATH.smoke-test-backup"
    cp "$CONFIG_PATH" "$CONFIG_BACKUP"
    log "backed up $CONFIG_PATH → $CONFIG_BACKUP"
  fi
  if [[ -d "$EXT_DIR" ]]; then
    EXT_BACKUP="$EXT_DIR.smoke-test-backup"
    rm -rf "$EXT_BACKUP"
    mv "$EXT_DIR" "$EXT_BACKUP"
    log "backed up $EXT_DIR → $EXT_BACKUP"
  fi

  # Start from a clean slate after backups are in place.
  log "clearing any pre-existing hindsight-openclaw install"
  yes 2>/dev/null | openclaw plugins uninstall hindsight-openclaw >/dev/null 2>&1 || true
  rm -rf "$EXT_DIR"

  # Pack the tarball unless one was provided.
  if [[ $# -gt 0 && -n "$1" ]]; then
    TARBALL="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
    log "using provided tarball: $TARBALL"
  else
    log "packing plugin tarball…"
    (
      cd "$PLUGIN_DIR"
      npm run clean --silent
      npm run build --silent
      npm pack --silent >/dev/null
    )
    TARBALL="$(ls -t "$PLUGIN_DIR"/vectorize-io-hindsight-openclaw-*.tgz | head -1)"
    log "packed: $TARBALL"
  fi

  # -------------------------------------------------------------------------
  # Phase 1 — tarball install (no --dangerously-force-unsafe-install)
  # -------------------------------------------------------------------------
  log "installing plugin from tarball (no --dangerously-force-unsafe-install)…"
  local install_log
  install_log="$(mktemp)"
  if ! openclaw plugins install "$TARBALL" >"$install_log" 2>&1; then
    cat "$install_log" >&2
    fail "openclaw plugins install failed"
  fi
  if grep -qi "dangerous code patterns detected" "$install_log"; then
    cat "$install_log" >&2
    fail "install scanner reported dangerous code patterns"
  fi
  if ! grep -q "Installed plugin: hindsight-openclaw" "$install_log"; then
    cat "$install_log" >&2
    fail "plugin install did not report success"
  fi
  rm -f "$install_log"
  log "✓ scanner-clean install succeeded"

  # Confirm deps resolved from npm.
  if [[ ! -d "$EXT_DIR/node_modules/@vectorize-io/hindsight-all" ]]; then
    fail "expected @vectorize-io/hindsight-all in installed extension"
  fi
  if [[ ! -d "$EXT_DIR/node_modules/@vectorize-io/hindsight-client" ]]; then
    fail "expected @vectorize-io/hindsight-client in installed extension"
  fi
  log "✓ workspace deps resolved from registry"

  [[ -f "$EXT_DIR/dist/setup.js" ]] || fail "setup.js missing in installed extension"

  # -------------------------------------------------------------------------
  # Phase 2 — non-interactive setup for each mode
  # -------------------------------------------------------------------------
  run_setup_mode "cloud (default URL, SecretRef)" \
    --mode cloud --token-env HINDSIGHT_CLOUD_TOKEN
  [[ "$(get_config_value plugins.entries.hindsight-openclaw.config.hindsightApiUrl)" == "https://api.hindsight.vectorize.io" ]] \
    || fail "cloud mode: hindsightApiUrl not set to the default URL"

  run_setup_mode "cloud (inline --token)" \
    --mode cloud --token hsk_smoke_test_inline_value
  # Read the raw file — `openclaw config get` redacts sensitive fields.
  [[ "$(get_raw_config_value plugins.entries.hindsight-openclaw.config.hindsightApiToken)" == "hsk_smoke_test_inline_value" ]] \
    || fail "cloud mode: --token did not roundtrip as inline string"

  run_setup_mode "external API (no auth)" \
    --mode api --api-url "$HINDSIGHT_API_URL" --no-token
  [[ "$(get_config_value plugins.entries.hindsight-openclaw.config.hindsightApiUrl)" == "$HINDSIGHT_API_URL" ]] \
    || fail "api mode: hindsightApiUrl did not roundtrip"

  run_setup_mode "external API (inline --token)" \
    --mode api --api-url "$HINDSIGHT_API_URL" --token api_smoke_test_inline
  [[ "$(get_raw_config_value plugins.entries.hindsight-openclaw.config.hindsightApiToken)" == "api_smoke_test_inline" ]] \
    || fail "api mode: --token did not roundtrip as inline string"

  run_setup_mode "embedded (openai --api-key-env + model override)" \
    --mode embedded --provider openai --api-key-env OPENAI_API_KEY --model gpt-4o-mini
  [[ "$(get_config_value plugins.entries.hindsight-openclaw.config.llmProvider)" == "openai" ]] \
    || fail "embedded mode: llmProvider not set to openai"
  [[ "$(get_config_value plugins.entries.hindsight-openclaw.config.llmModel)" == "gpt-4o-mini" ]] \
    || fail "embedded mode: llmModel not set to gpt-4o-mini"

  run_setup_mode "embedded (openai --api-key inline)" \
    --mode embedded --provider openai --api-key sk-smoke-test-inline
  [[ "$(get_raw_config_value plugins.entries.hindsight-openclaw.config.llmApiKey)" == "sk-smoke-test-inline" ]] \
    || fail "embedded mode: --api-key did not roundtrip as inline string"

  run_setup_mode "embedded (claude-code, no key)" \
    --mode embedded --provider claude-code

  # -------------------------------------------------------------------------
  # Phase 3 — negative tests: bad CLI args must fail fast
  # -------------------------------------------------------------------------
  log "verifying setup wizard rejects bad args…"
  if node "$EXT_DIR/dist/setup.js" --config-path "$CONFIG_PATH" --mode cloud 2>/dev/null; then
    fail "cloud mode without --token or --token-env should have failed"
  fi
  if node "$EXT_DIR/dist/setup.js" --config-path "$CONFIG_PATH" --mode cloud --token hsk_x --token-env T 2>/dev/null; then
    fail "cloud mode with both --token and --token-env should have failed"
  fi
  if node "$EXT_DIR/dist/setup.js" --config-path "$CONFIG_PATH" --mode api --api-url "$HINDSIGHT_API_URL" --token-env TOK --no-token 2>/dev/null; then
    fail "--token-env + --no-token together should have failed"
  fi
  if node "$EXT_DIR/dist/setup.js" --config-path "$CONFIG_PATH" --mode embedded --provider openai 2>/dev/null; then
    fail "openai without --api-key or --api-key-env should have failed"
  fi
  if node "$EXT_DIR/dist/setup.js" --config-path "$CONFIG_PATH" --mode embedded --provider openai --api-key sk-x --api-key-env OPENAI_API_KEY 2>/dev/null; then
    fail "embedded with both --api-key and --api-key-env should have failed"
  fi
  log "✓ bad args rejected"

  log "🎉 all smoke tests passed"
}

main "$@"
