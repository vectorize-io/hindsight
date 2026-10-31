#!/bin/bash
# Parallel linting for all code (Node, Python)
# Runs all linting tasks concurrently for faster execution.
#
# Locally it lints only the packages that differ from main; CI and LINT_ALL=1
# lint everything.
#
# Speed notes: every program start is expensive on some machines, so the
# script avoids helper commands (cat, grep, basename, dirname) on its
# sequential path, calls tool binaries directly instead of through npx / uv run
# / npm run, and lets eslint and prettier skip files unchanged since the last run.

set -e

SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
[ "$SCRIPT_DIR" = "${BASH_SOURCE[0]}" ] && SCRIPT_DIR=.
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
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

# Locally, lint only the packages that differ from main (committed on this
# branch, uncommitted, or untracked): main itself passed this same script in CI.
# CI, LINT_ALL=1, or a change to shared lint config lints everything.
# The two git lookups run side by side: each can take a second to start on macOS.
# (Integrations used to be compared against HEAD only, and everything else was
# always linted.)
INTEGRATIONS_DIR="$REPO_ROOT/hindsight-integrations"
if [ -n "$CI" ] || [ -n "$LINT_ALL" ] || [ -n "$LINT_ALL_INTEGRATIONS" ]; then
    LINT_ALL=1
else
    LINT_ALL=0
    git -C "$REPO_ROOT" diff --name-only --merge-base origin/main > "$TEMP_DIR/changed-tracked" 2> /dev/null &
    GIT_DIFF_PID=$!
    git -C "$REPO_ROOT" ls-files --others --exclude-standard > "$TEMP_DIR/changed-untracked" &
    GIT_LS_PID=$!
fi

CHANGED_FILES=""
if [ "$LINT_ALL" = "0" ]; then
    # No origin/main to compare with (fresh clone, other remote name): lint everything.
    if ! wait "$GIT_DIFF_PID"; then
        LINT_ALL=1
    fi
    wait "$GIT_LS_PID"
    CHANGED_FILES=$'\n'"$(< "$TEMP_DIR/changed-tracked")"$'\n'"$(< "$TEMP_DIR/changed-untracked")"$'\n'
fi

# True when any of the given paths (a directory ending in "/", or a file) changed.
changed() {
    [ "$LINT_ALL" = "1" ] && return 0
    local path
    for path in "$@"; do
        if [[ $path == */ ]]; then
            [[ $CHANGED_FILES == *$'\n'"$path"* ]] && return 0
        else
            [[ $CHANGED_FILES == *$'\n'"$path"$'\n'* ]] && return 0
        fi
    done
    return 1
}

# Shared lint setup: tool versions, configs, this script.
if changed uv.lock package-lock.json pyproject.toml package.json .python-version ruff.toml .prettierrc.json .gitignore scripts/hooks/lint.sh; then
    LINT_ALL=1
fi

# The Python packages import each other (hindsight-dev and hindsight-embed use
# hindsight_api), so a change to any of them type-checks all three.
PYTHON_PACKAGES=(hindsight-api-slim/ hindsight-api/ hindsight-dev/ hindsight-embed/ hindsight-clients/python/)
if changed "${PYTHON_PACKAGES[@]}" hindsight-integrations/; then
    echo "  Syncing Python dependencies..."
    # Run uv sync first so the environment is complete before the tools run.
    # Keep pre-commit from rewriting uv.lock. Dependency changes should update
    # the lockfile explicitly before linting.
    uv sync --frozen --quiet
    # Call the synced tools directly. Each task used to go through `uv run`, but
    # every `uv run` takes the same env lock, so the parallel tasks queued behind
    # each other. Node tools used to go through `npx`, which takes seconds to start.
    # VIRTUAL_ENV is what `uv run` sets: ty reads it to find the installed packages.
    export VIRTUAL_ENV="$REPO_ROOT/.venv"
    VENV_BIN="$VIRTUAL_ENV/bin"
fi

echo "  Running lints in parallel..."

# Node/TypeScript tasks
CP_DIR="$REPO_ROOT/hindsight-control-plane"
# eslint runs the package's own "lint" script, so CI keeps that script working.
# It used to go through `npm run lint`; npm alone takes over a second to start,
# so the script's command runs here the way npm runs it, with node_modules/.bin
# on PATH.
# With --fix, eslint re-lints every file that has warnings and ignores its cache,
# so check from the cache first and only run --fix when there is something to fix.
# "potentially fixable" is the summary line of eslint's default formatter.
eslint_task() {
    local package_json lint_script
    package_json="$(< package.json)"
    if [[ ! $package_json =~ \"lint\"[[:space:]]*:[[:space:]]*\"([^\"]*)\" ]]; then
        echo "hindsight-control-plane/package.json has no \"lint\" script"
        return 1
    fi
    lint_script="${BASH_REMATCH[1]}"
    local PATH="$PWD/node_modules/.bin:$REPO_ROOT/node_modules/.bin:$PATH"
    # Content, not mtime: a branch switch rewrites mtimes but rarely the content.
    local args="--cache --cache-strategy content --cache-location $CACHE_DIR/eslint/"
    local out
    if out=$(eval "$lint_script $args" 2>&1) && [[ $out != *"potentially fixable"* ]]; then
        echo "$out"
        return 0
    fi
    eval "$lint_script --fix $args"
}
if changed hindsight-control-plane/; then
    run_task "eslint" "$CP_DIR" "eslint_task"
    run_task "prettier" "$CP_DIR" "$(node_bin prettier "$CP_DIR") --write --cache --cache-location $CACHE_DIR/prettier/control-plane src/**/*.{ts,tsx}"
fi
if changed hindsight-clients/typescript/; then
    run_task "prettier-ts-client" "$REPO_ROOT/hindsight-clients/typescript" "$(prettier_cmd "$REPO_ROOT/hindsight-clients/typescript")"
fi
if changed hindsight-all-npm/; then
    run_task "prettier-all-npm" "$REPO_ROOT/hindsight-all-npm" "$(prettier_cmd "$REPO_ROOT/hindsight-all-npm")"
fi

if changed "${PYTHON_PACKAGES[@]}"; then
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
fi

# Python integrations use shared ruff.toml; Node integrations use shared .prettierrc.json.
if [ -d "$INTEGRATIONS_DIR" ] && changed hindsight-integrations/; then
    for dir in "$INTEGRATIONS_DIR"/*/; do
        dir="${dir%/}"
        name="${dir##*/}"
        changed "hindsight-integrations/$name/" || continue

        if [ -f "$dir/pyproject.toml" ]; then
            run_task "ruff-int-$name-check" "$dir" "$VENV_BIN/ruff check --fix --config $REPO_ROOT/ruff.toml ."
            run_task "ruff-int-$name-format" "$dir" "$VENV_BIN/ruff format --config $REPO_ROOT/ruff.toml ."
        elif [ -f "$dir/package.json" ]; then
            run_task "prettier-int-$name" "$dir" "$(prettier_cmd "$dir")"
        fi
    done
fi

if [ ${#PIDS[@]} -eq 0 ]; then
    echo "  No linted package differs from main, nothing to do (LINT_ALL=1 lints everything) ✓"
    exit 0
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
