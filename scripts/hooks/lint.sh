#!/bin/bash
# Parallel linting for all code (Node, Python)
# Runs all linting tasks concurrently for faster execution.
#
# Speed notes: every program start is expensive on some machines, so the
# script avoids helper commands (cat, grep, basename, git) on its sequential
# path, calls tool binaries directly instead of through npx / uv run, and lets
# eslint and prettier skip files unchanged since the last run.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Track all background jobs
declare -a PIDS
declare -a NAMES

run_task() {
    local name="$1"
    local dir="$2"
    shift 2
    local cmd="$@"

    # The subshell's exit code is the task's; its output is kept for failures.
    (
        cd "$dir"
        $cmd > "$TEMP_DIR/$name.output" 2>&1
    ) &
    PIDS+=($!)
    NAMES+=("$name")
}

# Find a Node tool the way npx does (nearest node_modules/.bin upward), but
# without npx's multi-second startup. Falls back to npx when it is not installed.
node_bin() {
    local tool="$1" dir="${2%/}"
    while [ -n "$dir" ]; do
        if [ -x "$dir/node_modules/.bin/$tool" ]; then
            echo "$dir/node_modules/.bin/$tool"
            return
        fi
        [ "$dir" = "$REPO_ROOT" ] && break
        dir="${dir%/*}"
    done
    echo "npx --yes $tool"
}

# eslint/prettier caches live under node_modules (gitignored); one per package.
CACHE_DIR="$REPO_ROOT/node_modules/.cache"
prettier_cmd() {
    local dir="${1%/}"
    echo "$(node_bin prettier "$dir") --write --cache --cache-location $CACHE_DIR/prettier/${dir##*/} --config $REPO_ROOT/.prettierrc.json --ignore-path $REPO_ROOT/.gitignore ."
}

# Integrations: lint packages with modifications vs HEAD locally; lint all in CI.
# The git lookups are slow in this repo, so they start now and run while the
# rest of the setup and the core lints do.
INTEGRATIONS_DIR="$REPO_ROOT/hindsight-integrations"
if [ -n "$CI" ] || [ -n "$LINT_ALL_INTEGRATIONS" ]; then
    LINT_ALL=1
else
    LINT_ALL=0
    { git -C "$REPO_ROOT" diff --name-only HEAD -- "hindsight-integrations/"; \
      git -C "$REPO_ROOT" ls-files --others --exclude-standard -- "hindsight-integrations/"; } > "$TEMP_DIR/changed" &
    GIT_PID=$!
fi

echo "  Syncing Python dependencies..."
# Run uv sync first so the environment is complete before the tools run.
# Keep pre-commit from rewriting uv.lock. Dependency changes should update
# the lockfile explicitly before linting.
uv sync --frozen --quiet
# Call the synced tools directly. Each task used to go through `uv run`, but
# every `uv run` takes the same env lock, so the parallel tasks queued behind
# each other. Node tools used to go through `npx`, which takes seconds to start.
VENV_BIN="$REPO_ROOT/.venv/bin"

echo "  Running lints in parallel..."

# Node/TypeScript tasks
CP_DIR="$REPO_ROOT/hindsight-control-plane"
# eslint goes through the package's own lint script, so CI keeps that script working.
# With --fix, eslint re-lints every file that has warnings and ignores its cache,
# so check from the cache first and only run --fix when there is something to fix.
# "potentially fixable" is the summary line of eslint's default formatter.
eslint_task() {
    local args=(--cache --cache-location "$CACHE_DIR/eslint/")
    local out
    if out=$(npm run lint -- "${args[@]}" 2>&1) && [[ $out != *"potentially fixable"* ]]; then
        echo "$out"
        return 0
    fi
    npm run lint -- --fix "${args[@]}"
}
run_task "eslint" "$CP_DIR" "eslint_task"
run_task "prettier" "$CP_DIR" "$(node_bin prettier "$CP_DIR") --write --cache --cache-location $CACHE_DIR/prettier/control-plane src/**/*.{ts,tsx}"
run_task "prettier-ts-client" "$REPO_ROOT/hindsight-clients/typescript" "$(prettier_cmd "$REPO_ROOT/hindsight-clients/typescript")"
run_task "prettier-all-npm" "$REPO_ROOT/hindsight-all-npm" "$(prettier_cmd "$REPO_ROOT/hindsight-all-npm")"

# Python hindsight-api-slim tasks
run_task "ruff-api-check" "$REPO_ROOT/hindsight-api-slim" "$VENV_BIN/ruff check --fix ."
run_task "ruff-api-format" "$REPO_ROOT/hindsight-api-slim" "$VENV_BIN/ruff format ."
run_task "ty-api" "$REPO_ROOT/hindsight-api-slim" "$VENV_BIN/ty check hindsight_api"

# Python hindsight-dev tasks
run_task "ruff-dev-check" "$REPO_ROOT/hindsight-dev" "$VENV_BIN/ruff check --fix ."
run_task "ruff-dev-format" "$REPO_ROOT/hindsight-dev" "$VENV_BIN/ruff format ."
run_task "ty-dev" "$REPO_ROOT/hindsight-dev" "$VENV_BIN/ty check hindsight_dev benchmarks"

# Python hindsight-embed tasks
run_task "ruff-embed-check" "$REPO_ROOT/hindsight-embed" "$VENV_BIN/ruff check --fix ."
run_task "ruff-embed-format" "$REPO_ROOT/hindsight-embed" "$VENV_BIN/ruff format ."
run_task "ty-embed" "$REPO_ROOT/hindsight-embed" "$VENV_BIN/ty check hindsight_embed"

# Python integrations use shared ruff.toml; Node integrations use shared .prettierrc.json.
CHANGED_FILES=""
if [ "$LINT_ALL" = "0" ]; then
    wait "$GIT_PID"
    CHANGED_FILES="$(< "$TEMP_DIR/changed")"
fi

integration_changed() {
    [ "$LINT_ALL" = "1" ] && return 0
    [[ $'\n'"$CHANGED_FILES" == *$'\n'"hindsight-integrations/$1/"* ]]
}

if [ -d "$INTEGRATIONS_DIR" ] && { [ "$LINT_ALL" = "1" ] || [ -n "$CHANGED_FILES" ]; }; then
    for dir in "$INTEGRATIONS_DIR"/*/; do
        dir="${dir%/}"
        name="${dir##*/}"
        integration_changed "$name" || continue

        if [ -f "$dir/pyproject.toml" ]; then
            run_task "ruff-int-$name-check" "$dir" "$VENV_BIN/ruff check --fix --config $REPO_ROOT/ruff.toml ."
            run_task "ruff-int-$name-format" "$dir" "$VENV_BIN/ruff format --config $REPO_ROOT/ruff.toml ."
        elif [ -f "$dir/package.json" ]; then
            run_task "prettier-int-$name" "$dir" "$(prettier_cmd "$dir")"
        fi
    done
fi

# Wait for all tasks and report the ones that failed
FAILED=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo ""
        echo "  ❌ ${NAMES[$i]} failed:"
        cat "$TEMP_DIR/${NAMES[$i]}.output"
        FAILED=1
    fi
done

if [ $FAILED -eq 1 ]; then
    exit 1
fi

echo "  All lints passed ✓"
