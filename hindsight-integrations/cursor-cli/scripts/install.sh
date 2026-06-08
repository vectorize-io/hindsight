#!/usr/bin/env bash
# Install the Hindsight Cursor CLI integration.
#
# Copies hook scripts to ~/.cursor/hooks/cursor-cli/ and merges a
# hooks block into ~/.cursor/hooks.json. Idempotent — safe to re-run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTEGRATION_ROOT="$(dirname "$SCRIPT_DIR")"
HOOKS_SRC="$INTEGRATION_ROOT/hooks/hooks.json"
SETTINGS_SRC="$INTEGRATION_ROOT/settings.json"

HOOKS_DIR="$HOME/.cursor/hooks/cursor-cli"
HOOKS_JSON="$HOME/.cursor/hooks.json"
SETTINGS_DST="$HOOKS_DIR/settings.json"
USER_CONFIG="$HOME/.hindsight/cursor-cli.json"

info() { printf '\033[0;32m[INFO]\033[0m %s\n' "$1"; }
warn() { printf '\033[0;33m[WARN]\033[0m %s\n' "$1"; }
err()  { printf '\033[0;31m[ERROR]\033[0m %s\n' "$1" >&2; }

if ! command -v python3 >/dev/null 2>&1; then
  err "python3 is not on PATH. Install Python 3.9+ and re-run."
  exit 1
fi

info "Installing Hindsight hook scripts to $HOOKS_DIR"
mkdir -p "$HOOKS_DIR"
cp -R "$INTEGRATION_ROOT/scripts/." "$HOOKS_DIR/scripts/"
cp "$SETTINGS_SRC" "$SETTINGS_DST"

# Render hooks.json with absolute paths to the installed scripts.
SCRIPTS_DIR="$HOOKS_DIR/scripts"
python3 -c "
import json, sys
src = json.load(open('$HOOKS_SRC'))
hooks = src.get('hooks', {})
for event, definitions in hooks.items():
    for definition in definitions:
        cmd = definition.get('command', '')
        if '__SCRIPTS_DIR__' in cmd:
            definition['command'] = cmd.replace('__SCRIPTS_DIR__', '$SCRIPTS_DIR')
print(json.dumps(src, indent=2))
" > "$HOOKS_DIR/hooks.json"

# Merge into ~/.cursor/hooks.json. We keep any pre-existing hooks the user
# already had configured.
if [ -f "$HOOKS_JSON" ]; then
  info "Merging into existing $HOOKS_JSON"
  python3 - "$HOOKS_JSON" "$HOOKS_DIR/hooks.json" <<'PY'
import json, sys
target, source = sys.argv[1], sys.argv[2]
try:
    existing = json.load(open(target))
except (OSError, ValueError):
    existing = {}
existing.setdefault("version", 1)
existing.setdefault("hooks", {})
with open(source) as f:
    new_block = json.load(f).get("hooks", {})
for event, definitions in new_block.items():
    bucket = existing["hooks"].setdefault(event, [])
    # Replace any existing hindsight entries (idempotent re-install).
    bucket = [d for d in bucket if "cursor-cli" not in json.dumps(d)]
    bucket.extend(definitions)
    existing["hooks"][event] = bucket
with open(target, "w") as f:
    json.dump(existing, f, indent=2)
PY
else
  info "Writing fresh $HOOKS_JSON"
  cp "$HOOKS_DIR/hooks.json" "$HOOKS_JSON"
fi

# Seed an empty user config the user can drop their token into.
if [ ! -f "$USER_CONFIG" ]; then
  info "Creating empty user config at $USER_CONFIG"
  mkdir -p "$(dirname "$USER_CONFIG")"
  cat > "$USER_CONFIG" <<'JSON'
{
  "hindsightApiUrl": "",
  "hindsightApiToken": null
}
JSON
else
  warn "User config already exists at $USER_CONFIG — leaving it alone"
fi

info "Done. Restart Cursor CLI to load the new hooks."
info "Logs (with debug=true): tail -F ~/.hindsight/cursor-cli/state/*.log"
