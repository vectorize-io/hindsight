#!/usr/bin/env bash
#
# Ensures every `hindsight-integrations/*/package-lock.json` resolves its
# dependencies from a public npm registry — not from a monorepo workspace
# symlink, a `file:` URL, or a relative path.
#
# Why this matters: integrations are published by `.github/workflows/
# release-integration.yml`, which runs `npm ci && npm run build` inside the
# integration directory. If the lockfile points at a workspace path whose
# `dist/` is gitignored and not pre-built in the release runner, tsc fails
# with `Cannot find module '@vectorize-io/...'`.
#
# This bit us once on integrations/openclaw/v0.6.0 — the lockfile had
# @vectorize-io/hindsight-client resolved to ../../hindsight-clients/typescript
# because `npm install` was originally run from the monorepo root, where npm
# silently preferred the workspace even though openclaw isn't itself listed
# in the root `workspaces` array. The test CI job masked it (it pre-builds
# workspace deps), but the release workflow doesn't.
#
# How to fix a flagged lockfile: from inside the integration directory,
#   rm -rf node_modules package-lock.json
#   npm install
# Never run `npm install` from the monorepo root for these integrations.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

FAIL=0

shopt -s nullglob
lockfiles=(hindsight-integrations/*/package-lock.json)
shopt -u nullglob

if [[ ${#lockfiles[@]} -eq 0 ]]; then
  echo -e "${YELLOW}[check-integration-lockfiles]${NC} no integration lockfiles found (nothing to check)"
  exit 0
fi

for lockfile in "${lockfiles[@]}"; do
  integration="$(basename "$(dirname "$lockfile")")"

  violations="$(python3 - "$lockfile" <<'PY'
import json, sys

path = sys.argv[1]
with open(path) as f:
    data = json.load(f)

bad = []
for name, info in data.get("packages", {}).items():
    # Only check actual node_modules entries. The root entry is "".
    if not name.startswith("node_modules/"):
        continue

    # Ignore optional deps that didn't install (they have no resolved URL
    # and no link path — they're legitimately absent for the current
    # platform).
    if info.get("optional") and "resolved" not in info and "link" not in info:
        continue

    resolved = info.get("resolved", "")
    link = info.get("link")

    if link is True:
        # Workspace symlink, no registry URL. This is the bug that broke
        # the openclaw 0.6.0 release — npm had resolved a registry-ranged
        # dep to a workspace symlink because `npm install` was run from
        # the monorepo root.
        bad.append(f"{name}: (link=true — workspace symlink)")
    elif not resolved:
        # Empty resolved on a non-link entry shouldn't happen, but flag
        # it anyway rather than silently accepting it.
        bad.append(f"{name}: (no `resolved` URL)")
    elif resolved.startswith("file:"):
        bad.append(f"{name}: {resolved}")
    elif resolved.startswith(("../", "./")):
        bad.append(f"{name}: {resolved}")

if bad:
    print("\n".join(bad))
PY
)"

  if [[ -n "$violations" ]]; then
    echo -e "${RED}[check-integration-lockfiles]${NC} $lockfile resolves dependencies outside the npm registry:"
    echo "$violations" | sed 's/^/    /'
    FAIL=1
  fi
done

if [[ $FAIL -eq 1 ]]; then
  cat >&2 <<'EOF'

How to fix a flagged lockfile: from inside the integration directory,
    rm -rf node_modules package-lock.json
    npm install

Do not run `npm install` from the monorepo root for these integrations
— it silently resolves `@vectorize-io/...` names to the workspace copy
even when the package.json declares a registry version range.
EOF
  exit 1
fi

echo -e "${GREEN}[check-integration-lockfiles]${NC} ✓ ${#lockfiles[@]} integration lockfile(s) resolve cleanly from the npm registry"
