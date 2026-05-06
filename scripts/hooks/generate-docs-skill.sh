#!/bin/bash
# Pre-commit hook: keep skills/hindsight-docs/ in sync with hindsight-docs/
# sources. CI's verify-generated-files job regenerates and fails the build on
# drift; this hook catches it locally first so PR authors don't ship a commit
# that needs a follow-up "regenerate docs skill" patch.

set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"

"$REPO_ROOT/scripts/generate-docs-skill.sh" >/dev/null

if ! git diff --quiet -- skills/hindsight-docs/; then
    echo ""
    echo "  ❌ skills/hindsight-docs/ was out of sync with hindsight-docs/ sources."
    echo "     The generator just refreshed it — stage the regen and re-commit:"
    echo "       git add skills/hindsight-docs/"
    echo ""
    git diff --stat -- skills/hindsight-docs/ | sed 's/^/         /'
    echo ""
    exit 1
fi
