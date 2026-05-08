#!/usr/bin/env bash
# Publish locomo benchmark results to the dashboard repo's gh-pages branch.
#
# Reads a locomo benchmark JSON, strips the heavy per-question detailed_results
# (kept only in the upload artifact), enriches with commit + workflow metadata,
# then pushes:
#   data/locomo/<timestamp>-<short_sha>.json
#   data/locomo-index.json   (manifest, newest first)
# to vectorize-io/hindsight-continuous-performance-monitor (gh-pages).
#
# Required env:
#   PERF_DASHBOARD_TOKEN   PAT with Contents:write on the dashboard repo
#
# Usage:
#   ./scripts/benchmarks/publish-locomo-results.sh path/to/benchmark_results.json

set -euo pipefail

INPUT_JSON="${1:?usage: $0 <benchmark_results.json>}"
DASHBOARD_REPO="${DASHBOARD_REPO:-vectorize-io/hindsight-continuous-performance-monitor}"
HINDSIGHT_REPO="${HINDSIGHT_REPO:-vectorize-io/hindsight}"

if [ ! -f "$INPUT_JSON" ]; then
  echo "Input JSON not found: $INPUT_JSON" >&2
  exit 1
fi
: "${PERF_DASHBOARD_TOKEN:?PERF_DASHBOARD_TOKEN must be set}"

# ─────────────────────────────────────────────────────────────────────────
# Capture commit + workflow metadata
# ─────────────────────────────────────────────────────────────────────────
SHA=$(git rev-parse HEAD)
SHORT_SHA=$(git rev-parse --short=8 HEAD)
SUBJECT=$(git log -1 --pretty=%s)
AUTHOR=$(git log -1 --pretty=%an)
AUTHOR_DATE=$(git log -1 --pretty=%aI)
COMMIT_URL="https://github.com/${HINDSIGHT_REPO}/commit/${SHA}"

PR_NUMBER=""
PR_URL=""
if command -v gh >/dev/null 2>&1; then
  PR_NUMBER=$(gh api "repos/${HINDSIGHT_REPO}/commits/${SHA}/pulls" \
    --jq '.[0].number // empty' 2>/dev/null || true)
  if [ -n "$PR_NUMBER" ]; then
    PR_URL="https://github.com/${HINDSIGHT_REPO}/pull/${PR_NUMBER}"
  fi
fi

RUN_ID="${GITHUB_RUN_ID:-}"
RUN_URL=""
if [ -n "$RUN_ID" ]; then
  RUN_REPO="${GITHUB_REPOSITORY:-$HINDSIGHT_REPO}"
  RUN_URL="https://github.com/${RUN_REPO}/actions/runs/${RUN_ID}"
fi

TIMESTAMP_FILE=$(date -u +%Y%m%dT%H%M%SZ)
TIMESTAMP_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)
DATA_FILE="data/locomo/${TIMESTAMP_FILE}-${SHORT_SHA}.json"

echo "Publishing locomo run for ${SHORT_SHA} → ${DATA_FILE}"

# ─────────────────────────────────────────────────────────────────────────
# Strip detailed_results (per-question detail) and enrich
#
# detailed_results is per-question {question, predicted_answer, reasoning,
# retrieved_memories, ...} and bloats each run to several MB. We keep only
# accuracy stats per item, which is enough for trend charts; the full data
# remains available as a workflow artifact.
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
  --arg timestamp "$TIMESTAMP_ISO" \
  '
    # Drop per-question payloads from each item.
    .item_results = (.item_results // [] | map(.metrics |= del(.detailed_results)))
    | . + {
        timestamp: $timestamp,
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
      }
  ' "$INPUT_JSON" > "$ENRICHED_TMP"

OVERALL=$(jq -r '.overall_accuracy' "$ENRICHED_TMP")
NUM_ITEMS=$(jq -r '.num_items // (.item_results | length)' "$ENRICHED_TMP")
TOTAL_QUESTIONS=$(jq -r '.total_questions // 0' "$ENRICHED_TMP")

# ─────────────────────────────────────────────────────────────────────────
# Clone dashboard repo's gh-pages branch
# ─────────────────────────────────────────────────────────────────────────
WORK=$(mktemp -d)
trap 'rm -f "$ENRICHED_TMP"; rm -rf "$WORK"' EXIT

git clone --quiet --depth 1 --branch gh-pages \
  "https://x-access-token:${PERF_DASHBOARD_TOKEN}@github.com/${DASHBOARD_REPO}.git" \
  "$WORK"

mkdir -p "$WORK/data/locomo"
cp "$ENRICHED_TMP" "$WORK/$DATA_FILE"

# ─────────────────────────────────────────────────────────────────────────
# Update manifest (data/locomo-index.json)
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
  --arg timestamp "$TIMESTAMP_ISO" \
  --argjson overall "$OVERALL" \
  --argjson num_items "$NUM_ITEMS" \
  --argjson total_questions "$TOTAL_QUESTIONS" \
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
    overall_accuracy: $overall,
    num_items: $num_items,
    total_questions: $total_questions
  }')

INDEX_FILE="$WORK/data/locomo-index.json"
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
git commit --quiet -m "locomo: add results for ${SHORT_SHA}"

if ! git push --quiet origin gh-pages; then
  echo "Push rejected, pulling and retrying..." >&2
  git pull --quiet --rebase origin gh-pages
  git push --quiet origin gh-pages
fi

echo "Published ${DATA_FILE} to ${DASHBOARD_REPO} gh-pages"
