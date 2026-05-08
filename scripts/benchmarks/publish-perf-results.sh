#!/usr/bin/env bash
# Publish perf-test results to the dashboard repo's gh-pages branch.
#
# Reads a perf-test --output JSON, enriches it with commit metadata, then
# pushes:
#   data/<timestamp>-<short_sha>.json   (the enriched run)
#   data/index.json                     (manifest, newest first)
# to vectorize-io/hindsight-continuous-performance-monitor (gh-pages).
#
# Required env:
#   PERF_DASHBOARD_TOKEN   PAT with Contents:write on the dashboard repo
#
# Usage:
#   ./scripts/benchmarks/publish-perf-results.sh path/to/perf-results.json

set -euo pipefail

INPUT_JSON="${1:?usage: $0 <perf-results.json>}"
DASHBOARD_REPO="${DASHBOARD_REPO:-vectorize-io/hindsight-continuous-performance-monitor}"
HINDSIGHT_REPO="${HINDSIGHT_REPO:-vectorize-io/hindsight}"

if [ ! -f "$INPUT_JSON" ]; then
  echo "Input JSON not found: $INPUT_JSON" >&2
  exit 1
fi
: "${PERF_DASHBOARD_TOKEN:?PERF_DASHBOARD_TOKEN must be set}"

# ─────────────────────────────────────────────────────────────────────────
# Capture commit metadata
# ─────────────────────────────────────────────────────────────────────────
SHA=$(git rev-parse HEAD)
SHORT_SHA=$(git rev-parse --short=8 HEAD)
SUBJECT=$(git log -1 --pretty=%s)
AUTHOR=$(git log -1 --pretty=%an)
AUTHOR_DATE=$(git log -1 --pretty=%aI)
COMMIT_URL="https://github.com/${HINDSIGHT_REPO}/commit/${SHA}"

# Best-effort PR lookup. May return empty for direct pushes or when gh isn't
# configured; we treat it as optional metadata.
PR_NUMBER=""
PR_URL=""
if command -v gh >/dev/null 2>&1; then
  PR_NUMBER=$(gh api "repos/${HINDSIGHT_REPO}/commits/${SHA}/pulls" \
    --jq '.[0].number // empty' 2>/dev/null || true)
  if [ -n "$PR_NUMBER" ]; then
    PR_URL="https://github.com/${HINDSIGHT_REPO}/pull/${PR_NUMBER}"
  fi
fi

# Workflow run URL — populated when running under GitHub Actions.
RUN_ID="${GITHUB_RUN_ID:-}"
RUN_URL=""
if [ -n "$RUN_ID" ]; then
  RUN_REPO="${GITHUB_REPOSITORY:-$HINDSIGHT_REPO}"
  RUN_URL="https://github.com/${RUN_REPO}/actions/runs/${RUN_ID}"
fi

TIMESTAMP_FILE=$(date -u +%Y%m%dT%H%M%SZ)
DATA_FILE="data/${TIMESTAMP_FILE}-${SHORT_SHA}.json"

echo "Publishing run for ${SHORT_SHA} → ${DATA_FILE}"

# ─────────────────────────────────────────────────────────────────────────
# Enrich the input JSON
# ─────────────────────────────────────────────────────────────────────────
ENRICHED_TMP=$(mktemp)
trap 'rm -f "$ENRICHED_TMP"' EXIT

jq \
  --arg sha "$SHA" \
  --arg short_sha "$SHORT_SHA" \
  --arg subject "$SUBJECT" \
  --arg author "$AUTHOR" \
  --arg author_date "$AUTHOR_DATE" \
  --arg commit_url "$COMMIT_URL" \
  --arg pr_number "$PR_NUMBER" \
  --arg pr_url "$PR_URL" \
  --arg run_id "$RUN_ID" \
  --arg run_url "$RUN_URL" \
  '. + {
    commit: {
      sha: $sha,
      short_sha: $short_sha,
      subject: $subject,
      author: $author,
      author_date: $author_date,
      url: $commit_url,
      pr_number: ($pr_number | if . == "" then null else tonumber end),
      pr_url: (if $pr_url == "" then null else $pr_url end)
    },
    workflow_run: (if $run_url == "" then null else {id: $run_id, url: $run_url} end)
  }' "$INPUT_JSON" > "$ENRICHED_TMP"

RUN_TIMESTAMP=$(jq -r '.timestamp' "$ENRICHED_TMP")
RUN_SCALE=$(jq -r '.scale' "$ENRICHED_TMP")

# ─────────────────────────────────────────────────────────────────────────
# Clone dashboard repo's gh-pages branch
# ─────────────────────────────────────────────────────────────────────────
WORK=$(mktemp -d)
trap 'rm -f "$ENRICHED_TMP"; rm -rf "$WORK"' EXIT

git clone --quiet --depth 1 --branch gh-pages \
  "https://x-access-token:${PERF_DASHBOARD_TOKEN}@github.com/${DASHBOARD_REPO}.git" \
  "$WORK"

mkdir -p "$WORK/data"
cp "$ENRICHED_TMP" "$WORK/$DATA_FILE"

# ─────────────────────────────────────────────────────────────────────────
# Update manifest (data/index.json)
# ─────────────────────────────────────────────────────────────────────────
NEW_ENTRY=$(jq -n \
  --arg sha "$SHA" \
  --arg short_sha "$SHORT_SHA" \
  --arg subject "$SUBJECT" \
  --arg author "$AUTHOR" \
  --arg author_date "$AUTHOR_DATE" \
  --arg commit_url "$COMMIT_URL" \
  --arg pr_url "$PR_URL" \
  --arg run_url "$RUN_URL" \
  --arg data_file "$DATA_FILE" \
  --arg timestamp "$RUN_TIMESTAMP" \
  --arg scale "$RUN_SCALE" \
  '{
    sha: $sha,
    short_sha: $short_sha,
    subject: $subject,
    author: $author,
    author_date: $author_date,
    commit_url: $commit_url,
    pr_url: (if $pr_url == "" then null else $pr_url end),
    run_url: (if $run_url == "" then null else $run_url end),
    data_file: $data_file,
    timestamp: $timestamp,
    scale: $scale
  }')

INDEX_FILE="$WORK/data/index.json"
if [ ! -f "$INDEX_FILE" ]; then
  echo '{"runs": []}' > "$INDEX_FILE"
fi

UPDATED_INDEX=$(jq \
  --argjson entry "$NEW_ENTRY" \
  '.runs = ([$entry] + (.runs // [])) | .updated_at = (now | todateiso8601)' \
  "$INDEX_FILE")
echo "$UPDATED_INDEX" > "$INDEX_FILE"

# ─────────────────────────────────────────────────────────────────────────
# Commit and push
# ─────────────────────────────────────────────────────────────────────────
cd "$WORK"
git config user.name 'hindsight-perf-bot'
git config user.email 'hindsight-perf-bot@users.noreply.github.com'
git add data/
if git diff --cached --quiet; then
  echo "No changes to commit (this shouldn't happen — skipping push)" >&2
  exit 0
fi
git commit --quiet -m "perf: add results for ${SHORT_SHA}"

# Retry push once if rejected — concurrent runs from different jobs could race.
if ! git push --quiet origin gh-pages; then
  echo "Push rejected, pulling and retrying..." >&2
  git pull --quiet --rebase origin gh-pages
  git push --quiet origin gh-pages
fi

echo "Published ${DATA_FILE} to ${DASHBOARD_REPO} gh-pages"
