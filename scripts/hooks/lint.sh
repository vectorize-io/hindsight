#!/bin/bash
# Parallel linting for all code (Node, Python)
# Runs all linting tasks concurrently for faster execution

set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
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

    (
        cd "$dir"
        if OUTPUT=$($cmd 2>&1); then
            echo "OK" > "$TEMP_DIR/$name.status"
        else
            echo "FAIL" > "$TEMP_DIR/$name.status"
            echo "$OUTPUT" > "$TEMP_DIR/$name.output"
        fi
    ) &
    PIDS+=($!)
    NAMES+=("$name")
}

echo "  Syncing Python dependencies..."
# Run uv sync first to avoid race conditions when multiple uv run commands
# try to reinstall local packages in parallel (e.g., after version bump)
uv sync --quiet

echo "  Running lints in parallel..."

# Node/TypeScript tasks
run_task "eslint" "$REPO_ROOT/hindsight-control-plane" "npx eslint --fix src/**/*.{ts,tsx}"
run_task "prettier" "$REPO_ROOT/hindsight-control-plane" "npx prettier --write src/**/*.{ts,tsx}"

# Python hindsight-api-slim tasks
run_task "ruff-api-check" "$REPO_ROOT/hindsight-api-slim" "uv run ruff check --fix ."
run_task "ruff-api-format" "$REPO_ROOT/hindsight-api-slim" "uv run ruff format ."
run_task "ty-api" "$REPO_ROOT/hindsight-api-slim" "uv run ty check hindsight_api"

# Python hindsight-dev tasks
run_task "ruff-dev-check" "$REPO_ROOT/hindsight-dev" "uv run ruff check --fix ."
run_task "ruff-dev-format" "$REPO_ROOT/hindsight-dev" "uv run ruff format ."
run_task "ty-dev" "$REPO_ROOT/hindsight-dev" "uv run ty check hindsight_dev benchmarks"

# Python hindsight-embed tasks
run_task "ruff-embed-check" "$REPO_ROOT/hindsight-embed" "uv run ruff check --fix ."
run_task "ruff-embed-format" "$REPO_ROOT/hindsight-embed" "uv run ruff format ."
run_task "ty-embed" "$REPO_ROOT/hindsight-embed" "uv run ty check hindsight_embed"

# Integrations: lint packages with modifications vs HEAD locally; lint all in CI.
# Python integrations use shared ruff.toml; Node integrations use shared .prettierrc.json.
INTEGRATIONS_DIR="$REPO_ROOT/hindsight-integrations"
if [ -n "$CI" ] || [ -n "$LINT_ALL_INTEGRATIONS" ]; then
    LINT_ALL=1
    CHANGED_FILES=""
else
    LINT_ALL=0
    CHANGED_FILES=$( { git -C "$REPO_ROOT" diff --name-only HEAD -- "hindsight-integrations/"; \
                       git -C "$REPO_ROOT" ls-files --others --exclude-standard -- "hindsight-integrations/"; } | sort -u )
fi

integration_changed() {
    [ "$LINT_ALL" = "1" ] && return 0
    local rel="hindsight-integrations/$1/"
    echo "$CHANGED_FILES" | grep -q "^$rel" && return 0 || return 1
}

if [ -d "$INTEGRATIONS_DIR" ] && { [ "$LINT_ALL" = "1" ] || [ -n "$CHANGED_FILES" ]; }; then
    for dir in "$INTEGRATIONS_DIR"/*/; do
        name=$(basename "$dir")
        integration_changed "$name" || continue

        if [ -f "$dir/pyproject.toml" ]; then
            run_task "ruff-int-$name-check" "$dir" "uv run --no-project ruff check --fix --config $REPO_ROOT/ruff.toml ."
            run_task "ruff-int-$name-format" "$dir" "uv run --no-project ruff format --config $REPO_ROOT/ruff.toml ."
        elif [ -f "$dir/package.json" ]; then
            run_task "prettier-int-$name" "$dir" "npx --yes prettier --write --config $REPO_ROOT/.prettierrc.json --ignore-path $REPO_ROOT/.gitignore ."
        fi
    done
fi

# Wait for all tasks to complete
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

# Check results
FAILED=0
for name in "${NAMES[@]}"; do
    if [ -f "$TEMP_DIR/$name.status" ]; then
        STATUS=$(cat "$TEMP_DIR/$name.status")
        if [ "$STATUS" = "FAIL" ]; then
            echo ""
            echo "  ❌ $name failed:"
            cat "$TEMP_DIR/$name.output"
            FAILED=1
        fi
    else
        echo "  ❌ $name: no status (crashed?)"
        FAILED=1
    fi
done

if [ $FAILED -eq 1 ]; then
    exit 1
fi

echo "  All lints passed ✓"
