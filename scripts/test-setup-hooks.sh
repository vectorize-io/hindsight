#!/bin/bash
# Regression test for scripts/setup-hooks.sh (#6230): the hooks path must be
# relative so each worktree runs its own hooks, and a custom path must survive.

set -euo pipefail

SCRIPT="$(cd "$(dirname "$0")" && pwd)/setup-hooks.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cd "$TMP"
git init -q main
cd main
git config user.email t@t && git config user.name t
mkdir .githooks
printf '#!/bin/sh\ntouch "$(git rev-parse --show-toplevel)/hook-ran"\n' > .githooks/pre-commit
chmod +x .githooks/pre-commit
git add . && git commit -qm init --no-verify
git worktree add -q ../wt

expect() { [ "$(git config --get core.hooksPath)" = "$1" ] || { echo "FAIL: expected '$1', got '$(git config --get core.hooksPath)'"; exit 1; }; }

# Unset -> relative.
(cd ../wt && bash "$SCRIPT" >/dev/null)
expect .githooks

# The hook runs from the worktree's own copy, not from whoever ran setup.
(cd ../wt && git commit -q --allow-empty -m x)
[ -f ../wt/hook-ran ] && [ ! -f hook-ran ] || { echo "FAIL: hook did not run in the worktree"; exit 1; }

# The absolute value older versions wrote -> rewritten to relative.
git config core.hooksPath /gone/worktree/.githooks
bash "$SCRIPT" >/dev/null
expect .githooks

# A developer's own path -> left alone.
git config core.hooksPath /my/custom/hooks
bash "$SCRIPT" >/dev/null
expect /my/custom/hooks

echo "setup-hooks.sh: all checks passed"
