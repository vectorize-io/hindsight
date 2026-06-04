#!/bin/bash
#
# One-shot dev environment setup for Hindsight.
#
# Installs every toolchain the repo needs (uv/Python, Node/npm, Rust/cargo),
# installs all workspace dependencies, and builds the core artifacts so the
# project is ready to develop with — including offline, after this finishes.
#
# Each step checks whether it's already done and skips it, so the script is
# safe to re-run. Docker image builds are intentionally out of scope.
#
# Usage:
#   ./scripts/dev/setup.sh              # install toolchains + deps + core builds
#   ./scripts/dev/setup.sh --skip-build  # install toolchains + deps only
#   ./scripts/dev/setup.sh --skip-models # don't pre-download local ML models
#   ./scripts/dev/setup.sh --with-docs   # also build the docs site
#   ./scripts/dev/setup.sh --force       # rebuild artifacts even if present
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# --- options ----------------------------------------------------------------
SKIP_BUILD=false
SKIP_MODELS=false
WITH_DOCS=false
FORCE=false
for arg in "$@"; do
    case "$arg" in
        --skip-build)  SKIP_BUILD=true ;;
        --skip-models) SKIP_MODELS=true ;;
        --with-docs)   WITH_DOCS=true ;;
        --force)       FORCE=true ;;
        -h|--help)
            sed -n '3,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown option: $arg (try --help)"; exit 2 ;;
    esac
done

# --- pretty logging ---------------------------------------------------------
step()  { printf '\n\033[1;34m▶ %s\033[0m\n' "$1"; }
ok()    { printf '  \033[32m✓\033[0m %s\n' "$1"; }
info()  { printf '  \033[2m• %s\033[0m\n' "$1"; }
warn()  { printf '  \033[33m⚠ %s\033[0m\n' "$1"; }
have()  { command -v "$1" >/dev/null 2>&1; }

OS="$(uname -s)"
SUMMARY=()

# Minimum Node major version the workspaces expect (CI builds on 20/22).
NODE_MIN_MAJOR=20

# ---------------------------------------------------------------------------
# Toolchains
# ---------------------------------------------------------------------------

ensure_uv() {
    step "uv (Python toolchain & package manager)"
    # uv installs to ~/.local/bin by default — make sure it's reachable now.
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if have uv; then
        ok "already installed ($(uv --version))"
        SUMMARY+=("uv: present")
        return
    fi
    info "installing via astral.sh installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer writes an env file with the PATH update; source it so the
    # rest of this run can use uv immediately.
    [ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
    export PATH="$HOME/.local/bin:$PATH"
    have uv || { warn "uv not on PATH after install — open a new shell and re-run"; exit 1; }
    ok "installed ($(uv --version))"
    SUMMARY+=("uv: installed")
}

node_major() { node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0; }

ensure_node() {
    step "Node.js + npm"
    if have node && [ "$(node_major)" -ge "$NODE_MIN_MAJOR" ]; then
        ok "already installed (node $(node --version), npm $(npm --version))"
        SUMMARY+=("node: present ($(node --version))")
        return
    fi
    if have node; then
        warn "node $(node --version) is older than v${NODE_MIN_MAJOR} — upgrading"
    fi

    # Pick the most appropriate installer for the host. Order matters: prefer a
    # system package manager, fall back to nvm (no root, works in containers).
    if [ "$OS" = "Darwin" ] && have brew; then
        info "installing via Homebrew..."
        brew install node
    elif have apt-get; then
        info "installing via apt (NodeSource ${NODE_MIN_MAJOR}.x)..."
        local SUDO=""
        [ "$(id -u)" -ne 0 ] && have sudo && SUDO="sudo"
        curl -fsSL "https://deb.nodesource.com/setup_${NODE_MIN_MAJOR}.x" | $SUDO bash -
        $SUDO apt-get install -y nodejs
    else
        info "installing via nvm..."
        export NVM_DIR="$HOME/.nvm"
        if [ ! -s "$NVM_DIR/nvm.sh" ]; then
            curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
        fi
        # shellcheck disable=SC1091
        . "$NVM_DIR/nvm.sh"
        nvm install "$NODE_MIN_MAJOR"
        nvm alias default "$NODE_MIN_MAJOR"
    fi

    have node || { warn "node not on PATH after install — open a new shell and re-run"; exit 1; }
    ok "installed (node $(node --version), npm $(npm --version))"
    SUMMARY+=("node: installed ($(node --version))")
}

ensure_rust() {
    step "Rust + cargo (for hindsight-cli)"
    [ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
    export PATH="$HOME/.cargo/bin:$PATH"
    if have cargo; then
        ok "already installed ($(cargo --version))"
        SUMMARY+=("cargo: present")
        return
    fi
    info "installing via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
    . "$HOME/.cargo/env"
    have cargo || { warn "cargo not on PATH after install — open a new shell and re-run"; exit 1; }
    ok "installed ($(cargo --version))"
    SUMMARY+=("cargo: installed")
}

# ---------------------------------------------------------------------------
# Repo bootstrap
# ---------------------------------------------------------------------------

ensure_env_file() {
    step ".env file"
    if [ -f "$ROOT_DIR/.env" ]; then
        ok ".env already exists"
        return
    fi
    cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
    ok "created .env from .env.example"
    warn "set HINDSIGHT_API_LLM_API_KEY (and provider/model) in .env before running the API"
    SUMMARY+=(".env: created — add your LLM API key")
}

setup_git_hooks() {
    step "Git hooks"
    if [ "$(git config --get core.hooksPath || true)" = "$ROOT_DIR/.githooks" ]; then
        ok "hooks already configured"
        return
    fi
    ./scripts/setup-hooks.sh >/dev/null
    ok "configured core.hooksPath -> .githooks"
}

# ---------------------------------------------------------------------------
# Dependencies (also primes offline caches)
# ---------------------------------------------------------------------------

install_python_deps() {
    step "Python dependencies (uv sync — whole workspace)"
    # Installs the pinned interpreter (.python-version) if needed plus every
    # workspace member's deps into .venv. Populates uv's cache for offline use.
    uv sync
    ok "Python workspace synced (.venv ready)"
    SUMMARY+=("python deps: synced")
}

install_node_deps() {
    step "Node dependencies (npm ci — all workspaces)"
    if [ -d "$ROOT_DIR/node_modules" ] && [ "$FORCE" = false ]; then
        # npm ci is the clean-install path; only skip the reinstall when the
        # tree already exists and the user didn't ask to force.
        info "node_modules present — running npm install to reconcile lockfile"
        npm install --no-audit --no-fund
    else
        npm ci --no-audit --no-fund
    fi
    ok "Node workspaces installed"
    SUMMARY+=("node deps: installed")
}

# ---------------------------------------------------------------------------
# ML model cache (offline readiness) — mirrors docker/standalone/Dockerfile
# ---------------------------------------------------------------------------

prewarm_models() {
    step "Pre-download ML models & tokenizer (offline readiness)"
    # Respect provider/model overrides from .env; default to local + repo
    # defaults (BAAI/bge-small-en-v1.5, cross-encoder/ms-marco-MiniLM-L-6-v2).
    set -a; [ -f "$ROOT_DIR/.env" ] && . "$ROOT_DIR/.env"; set +a

    # Only the "local" provider downloads weights; remote providers (TEI,
    # OpenAI, Cohere, ...) fetch nothing here. tiktoken is always cached
    # because token counting runs regardless of provider.
    local emb_model="" rer_model=""
    if [ "${HINDSIGHT_API_EMBEDDINGS_PROVIDER:-local}" = "local" ]; then
        emb_model="${HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL:-BAAI/bge-small-en-v1.5}"
        info "embeddings (local): $emb_model"
    else
        info "embeddings provider is '${HINDSIGHT_API_EMBEDDINGS_PROVIDER}' — no local weights to fetch"
    fi
    if [ "${HINDSIGHT_API_RERANKER_PROVIDER:-local}" = "local" ]; then
        rer_model="${HINDSIGHT_API_RERANKER_LOCAL_MODEL:-cross-encoder/ms-marco-MiniLM-L-6-v2}"
        info "reranker (local): $rer_model"
    else
        info "reranker provider is '${HINDSIGHT_API_RERANKER_PROVIDER}' — no local weights to fetch"
    fi

    # Retry with exponential backoff for transient network failures, matching
    # the Docker build. Models land in the shared HF hub cache, so a re-run is
    # a fast cache validation rather than a re-download.
    local attempt delay=10
    for attempt in 1 2 3; do
        if PREWARM_EMB="$emb_model" PREWARM_RER="$rer_model" \
           HF_HUB_DOWNLOAD_TIMEOUT=600 \
           uv run --directory "$ROOT_DIR/hindsight-api-slim" python - <<'PY'
import os

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

import tiktoken

print("  caching tiktoken cl100k_base encoding...", flush=True)
tiktoken.get_encoding("cl100k_base")

emb = os.environ.get("PREWARM_EMB") or ""
rer = os.environ.get("PREWARM_RER") or ""
if emb:
    from sentence_transformers import SentenceTransformer

    print(f"  caching embedding model {emb} ...", flush=True)
    SentenceTransformer(emb)
if rer:
    from sentence_transformers import CrossEncoder

    print(f"  caching cross-encoder model {rer} ...", flush=True)
    CrossEncoder(rer)
print("  models cached", flush=True)
PY
        then
            ok "model cache warmed"
            SUMMARY+=("models: cached for offline use")
            return
        fi
        if [ "$attempt" -lt 3 ]; then
            warn "download attempt $attempt failed — retrying in ${delay}s"
            sleep "$delay"
            delay=$((delay * 2))
        fi
    done
    warn "model pre-download failed after 3 attempts (network?) — models will download on first API use"
    SUMMARY+=("models: NOT cached (will download at runtime)")
}

# ---------------------------------------------------------------------------
# Builds
# ---------------------------------------------------------------------------

build_ts_client() {
    step "Build TypeScript SDK (@vectorize-io/hindsight-client)"
    # The control plane imports the built SDK, so this must come first.
    if [ "$FORCE" = false ] && [ -d "$ROOT_DIR/hindsight-clients/typescript/dist" ]; then
        ok "dist/ present — skipping (use --force to rebuild)"
        return
    fi
    npm run build -w @vectorize-io/hindsight-client
    ok "SDK built"
    SUMMARY+=("ts-client: built")
}

build_cli() {
    step "Build Hindsight CLI (cargo release)"
    # Building also vendors every crate into ~/.cargo, priming the offline cache.
    if [ "$FORCE" = false ] && [ -x "$ROOT_DIR/hindsight-cli/target/release/hindsight" ]; then
        ok "release binary present — skipping (use --force to rebuild)"
        return
    fi
    ( cd "$ROOT_DIR/hindsight-cli" && cargo build --release )
    ok "CLI built (hindsight-cli/target/release/hindsight)"
    SUMMARY+=("cli: built")
}

build_docs() {
    step "Build documentation site (Docusaurus)"
    INCLUDE_CURRENT_VERSION=true npm run build -w hindsight-docs
    ok "Docs built"
    SUMMARY+=("docs: built")
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

printf '\033[1mHindsight dev setup\033[0m  (root: %s)\n' "$ROOT_DIR"
[ "$SKIP_BUILD" = true ] && info "build steps disabled (--skip-build)"
[ "$SKIP_MODELS" = true ] && info "model pre-download disabled (--skip-models)"
[ "$FORCE" = true ] && info "force rebuild enabled (--force)"

ensure_uv
ensure_node
ensure_rust
ensure_env_file
setup_git_hooks

install_python_deps
install_node_deps

[ "$SKIP_MODELS" = false ] && prewarm_models

if [ "$SKIP_BUILD" = false ]; then
    # NB: the control plane's Next.js production/standalone build is the Docker
    # deploy artifact and is intentionally NOT built here. Local dev runs it via
    # `npm run dev` (Turbopack, compiles on demand, no network needed offline).
    build_ts_client
    build_cli
    [ "$WITH_DOCS" = true ] && build_docs
fi

step "Setup complete"
for line in "${SUMMARY[@]}"; do ok "$line"; done
cat <<'EOF'

Next steps:
  • Add your LLM API key to .env (HINDSIGHT_API_LLM_API_KEY)
  • Start everything:        ./scripts/dev/start.sh
  • API only:                ./scripts/dev/start-api.sh
  • Run API tests:           cd hindsight-api-slim && uv run pytest tests/

The local embedding/reranker models and tiktoken encoding are cached, so the
API can run fully offline (default local providers). Re-run with --skip-models
to skip that step.
EOF
