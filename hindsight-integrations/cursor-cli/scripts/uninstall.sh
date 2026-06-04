#!/usr/bin/env bash
# Uninstall the Hindsight Cursor CLI integration.
#
# Removes the hook scripts under ~/.cursor/hooks/cursor-cli/ and
# strips Hindsight's hook entries from ~/.cursor/hooks.json.
# Leaves the user config (~/.hindsight/cursor-cli.json) intact.

set -euo pipefail

HOOKS_DIR="$HOME/.cursor/hooks/cursor-cli"
HOOKS_JSON="$HOME/.cursor/hooks.json"

info() { printf '\033[0;32m[INFO]\033[0m %s\n' "$1"; }
warn() { printf '\033[0;33m[WARN]\033[0m %s\n' "$1"; }

if [ -d "$HOOKS_DIR" ]; then
  info "Removing $HOOKS_DIR"
  rm -rf "$HOOKS_DIR"
else
  warn "$HOOKS_DIR does not exist — nothing to remove"
fi

if [ -f "$HOOKS_JSON" ]; then
  info "Stripping Hindsight entries from $HOOKS_JSON"
  python3 - "$HOOKS_JSON" <<'PY'
import json, sys
target = sys.argv[1]
try:
    data = json.load(open(target))
except (OSError, ValueError):
    sys.exit(0)
hooks = data.get("hooks", {})
for event, definitions in list(hooks.items()):
    kept = [d for d in definitions if "cursor-cli" not in json.dumps(d)]
    if kept:
        hooks[event] = kept
    else:
        del hooks[event]
data["hooks"] = hooks
with open(target, "w") as f:
    json.dump(data, f, indent=2)
PY
else
  warn "$HOOKS_JSON does not exist — nothing to strip"
fi

info "Done. Restart Cursor CLI to unload the hooks."
info "User config at ~/.hindsight/cursor-cli.json was preserved."
