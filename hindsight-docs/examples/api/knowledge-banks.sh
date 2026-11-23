#!/bin/bash
# Knowledge Banks examples for Hindsight CLI
# Run: bash examples/api/knowledge-banks.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The upload example reads its files from the current directory
WORKDIR="$(mktemp -d)"
cd "$WORKDIR"
cp "$SCRIPT_DIR/sample.pdf" x200-manual.pdf
printf '# X200 quick start\n\nFill the tank, then hold the power button for two seconds.\n' > x200-quickstart.md

# =============================================================================
# Setup (not shown in docs)
# =============================================================================
hindsight bank create support-agent --name "Support Agent" > /dev/null

# =============================================================================
# Doc Examples
# =============================================================================

# [docs:create-knowledge-bank]
hindsight kb create product-manuals
# [/docs:create-knowledge-bank]

# [docs:ingest-files]
# Each file becomes one document; --wait returns once they are searchable.
# Without --document-id, a file's id is its path (here: x200-manual.pdf),
# and a folder uploads every file in it.
hindsight kb ingest product-manuals x200-manual.pdf \
  --document-id x200-manual --tags x200 --wait
hindsight kb ingest product-manuals x200-quickstart.md \
  --document-id x200-quickstart --tags x200,quickstart --wait
# [/docs:ingest-files]

# [docs:ingest-text]
hindsight kb ingest-text product-manuals \
  "# Returns

You can return an X200 within 30 days..." \
  --document-id returns-policy --tags policy
# [/docs:ingest-text]

# [docs:search]
hindsight kb search product-manuals "How long do I have to return an X200?" --max-tokens 2048
# [/docs:search]

# [docs:list-tags]
hindsight kb list-tags product-manuals -q 'x2*'
# [/docs:list-tags]

# [docs:documents]
hindsight kb list-documents product-manuals --status ready
hindsight kb get-document product-manuals x200-manual
hindsight kb download product-manuals x200-manual --out manual-copy.pdf
hindsight kb passage-media product-manuals x200-manual 0 --out page-1.png
# [/docs:documents]

OPERATION_ID=$(hindsight kb operation list product-manuals -o json | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1)

# [docs:operations]
hindsight kb operation list product-manuals
hindsight kb operation get product-manuals "$OPERATION_ID"
# [/docs:operations]

# [docs:link-memory-bank]
hindsight bank set-config support-agent --knowledge-banks product-manuals
# [/docs:link-memory-bank]

# [docs:bank-config]
hindsight kb set-config product-manuals \
  --set text_passage_tokens=256 \
  --set 'disabled_media_types=["video"]'
# [/docs:bank-config]

# =============================================================================
# Checks (not shown in docs): the calls above must have worked, not just answered
# =============================================================================
hindsight kb search product-manuals "return an X200 within 30 days" -o json | grep -q '"returns-policy"'
hindsight kb list-tags product-manuals -o json | grep -q '"policy"'
hindsight kb get-document product-manuals x200-quickstart -o json | grep -q '"ready"'
cmp "$WORKDIR/x200-manual.pdf" manual-copy.pdf
test -s page-1.png
hindsight bank config support-agent -o json | grep -q '"product-manuals"'
hindsight kb config product-manuals --overrides-only -o json | grep -q '"text_passage_tokens": *256'

# [docs:settings]
# Every setting as the bank uses it, then only what the bank changes
hindsight kb config product-manuals
hindsight kb config product-manuals --overrides-only

# Drop one override, or all of them
hindsight kb set-config product-manuals --unset disabled_media_types
hindsight kb reset-config product-manuals -y
# [/docs:settings]
hindsight kb config product-manuals --overrides-only -o json | grep -q '^{}$'

# [docs:delete-document]
hindsight kb delete-document product-manuals returns-policy
# [/docs:delete-document]

# [docs:delete-knowledge-bank]
hindsight kb delete product-manuals -y
# [/docs:delete-knowledge-bank]

# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
rm -rf "$WORKDIR"
hindsight bank delete support-agent -y > /dev/null

echo "knowledge-banks.sh: All examples passed"
