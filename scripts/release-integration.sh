#!/bin/bash
set -e

cd "$(dirname "$0")/.."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

VALID_INTEGRATIONS=("litellm" "pydantic-ai" "crewai" "ag2" "ai-sdk" "chat" "openclaw" "langgraph" "llamaindex" "nemoclaw" "strands" "claude-code" "codex" "hermes" "autogen" "paperclip" "opencode" "cloudflare-oauth-proxy" "openai-agents" "pipecat" "agentcore" "smolagents" "n8n")

usage() {
    print_error "Usage: $0 <integration> <version>"
    echo ""
    echo "  integration  One of: ${VALID_INTEGRATIONS[*]}"
    echo "  version      Semantic version (e.g. 0.2.0) or bump keyword: patch, minor, major"
    echo ""
    echo "Examples:"
    echo "  $0 litellm 0.2.0"
    echo "  $0 pydantic-ai patch"
    echo "  $0 crewai minor"
    exit 1
}

if [ -z "$1" ] || [ -z "$2" ]; then
    usage
fi

INTEGRATION=$1
VERSION_ARG=$2

# Validate integration name
VALID=false
for v in "${VALID_INTEGRATIONS[@]}"; do
    if [ "$v" = "$INTEGRATION" ]; then
        VALID=true
        break
    fi
done
if [ "$VALID" = "false" ]; then
    print_error "Unknown integration '$INTEGRATION'"
    print_info "Valid integrations: ${VALID_INTEGRATIONS[*]}"
    exit 1
fi

# Read current version from package manifest
get_current_version() {
    local dir="hindsight-integrations/$INTEGRATION"
    if [ -f "$dir/pyproject.toml" ]; then
        grep '^version = ' "$dir/pyproject.toml" | sed 's/version = "\(.*\)"/\1/'
    elif [ -f "$dir/package.json" ]; then
        grep '"version"' "$dir/package.json" | head -1 | sed 's/.*"version": "\(.*\)".*/\1/'
    elif [ -f "$dir/.claude-plugin/plugin.json" ]; then
        grep '"version"' "$dir/.claude-plugin/plugin.json" | head -1 | sed 's/.*"version": "\(.*\)".*/\1/'
    elif [ -f "$dir/settings.json" ] && grep -q '"version"' "$dir/settings.json"; then
        grep '"version"' "$dir/settings.json" | head -1 | sed 's/.*"version": "\(.*\)".*/\1/'
    else
        echo ""
    fi
}

# Bump a semver component: bump_version <current> <part>
bump_version() {
    local current=$1 part=$2
    local major minor patch
    IFS='.' read -r major minor patch <<< "$current"
    case "$part" in
        major) echo "$((major + 1)).0.0" ;;
        minor) echo "$major.$((minor + 1)).0" ;;
        patch) echo "$major.$minor.$((patch + 1))" ;;
    esac
}

# Resolve version: either an explicit semver or a bump keyword
if [[ "$VERSION_ARG" =~ ^(patch|minor|major)$ ]]; then
    CURRENT_VERSION=$(get_current_version)
    if [ -z "$CURRENT_VERSION" ]; then
        print_error "Could not read current version for '$INTEGRATION'"
        exit 1
    fi
    VERSION=$(bump_version "$CURRENT_VERSION" "$VERSION_ARG")
    print_info "Bumping $CURRENT_VERSION → $VERSION ($VERSION_ARG)"
elif [[ "$VERSION_ARG" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    VERSION=$VERSION_ARG
else
    print_error "Invalid version: '$VERSION_ARG'. Use a semver (e.g. 0.2.0) or bump keyword (patch, minor, major)"
    exit 1
fi

TAG="integrations/$INTEGRATION/v$VERSION"

print_info "Releasing $INTEGRATION v$VERSION (tag: $TAG)"

# Check if we're on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    print_warn "You are not on the main branch (current: $CURRENT_BRANCH)"
    read -p "Do you want to continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_error "Release cancelled"
        exit 1
    fi
fi

# Check if working directory is clean
if [[ -n $(git status -s) ]]; then
    print_error "Working directory is not clean. Please commit or stash your changes."
    git status -s
    exit 1
fi

# Check if tag already exists
if git rev-parse "$TAG" >/dev/null 2>&1; then
    print_error "Tag $TAG already exists"
    exit 1
fi

# Load .env for OPENAI_API_KEY if needed
if [ -z "$OPENAI_API_KEY" ] && [ -f ".env" ]; then
    print_info "Loading environment from .env"
    set -a
    source ".env"
    set +a
fi

if [ -z "$OPENAI_API_KEY" ]; then
    print_error "OPENAI_API_KEY is not set and no .env file found. Required for changelog generation."
    exit 1
fi

# Determine integration type and update version
INTEGRATION_DIR="hindsight-integrations/$INTEGRATION"

if [ ! -d "$INTEGRATION_DIR" ]; then
    print_error "Integration directory not found: $INTEGRATION_DIR"
    exit 1
fi

if [ -f "$INTEGRATION_DIR/pyproject.toml" ]; then
    print_info "Updating version in $INTEGRATION_DIR/pyproject.toml"
    sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" "$INTEGRATION_DIR/pyproject.toml"
    rm "$INTEGRATION_DIR/pyproject.toml.bak"
elif [ -f "$INTEGRATION_DIR/package.json" ]; then
    print_info "Updating version in $INTEGRATION_DIR/package.json"
    sed -i.bak "s/\"version\": \".*\"/\"version\": \"$VERSION\"/" "$INTEGRATION_DIR/package.json"
    rm "$INTEGRATION_DIR/package.json.bak"
elif [ -f "$INTEGRATION_DIR/.claude-plugin/plugin.json" ]; then
    print_info "Updating version in $INTEGRATION_DIR/.claude-plugin/plugin.json"
    sed -i.bak "s/\"version\": \".*\"/\"version\": \"$VERSION\"/" "$INTEGRATION_DIR/.claude-plugin/plugin.json"
    rm "$INTEGRATION_DIR/.claude-plugin/plugin.json.bak"
elif [ -f "$INTEGRATION_DIR/settings.json" ] && grep -q '"version"' "$INTEGRATION_DIR/settings.json"; then
    print_info "Updating version in $INTEGRATION_DIR/settings.json"
    sed -i.bak "s/\"version\": \".*\"/\"version\": \"$VERSION\"/" "$INTEGRATION_DIR/settings.json"
    rm "$INTEGRATION_DIR/settings.json.bak"
else
    print_error "No pyproject.toml, package.json, plugin.json, or versioned settings.json found in $INTEGRATION_DIR"
    exit 1
fi

# Generate changelog entry using LLM
print_info "Generating changelog entry..."
if cd hindsight-dev && uv run generate-changelog "$VERSION" --integration "$INTEGRATION"; then
    cd ..
    print_info "Changelog generated"
else
    cd ..
    print_error "Changelog generation failed"
    git checkout .
    exit 1
fi

# Regenerate docs skill so changelog/SDK pages stay in sync
print_info "Regenerating docs skill..."
./scripts/generate-docs-skill.sh

# Commit version bump + changelog + regenerated skill together
print_info "Committing changes..."
git add "hindsight-integrations/$INTEGRATION/" "hindsight-docs/src/pages/changelog/integrations/$INTEGRATION.md" "skills/"
git commit --no-verify -m "release($INTEGRATION): v$VERSION"

# Create annotated tag
print_info "Creating tag $TAG..."
git tag -a "$TAG" -m "Release $INTEGRATION v$VERSION"

# Push commit and tag
print_info "Pushing to remote..."
git push origin "$CURRENT_BRANCH"
git push origin "$TAG"

print_info "✅ Released $INTEGRATION v$VERSION"
print_info "Tag: $TAG"
print_info "GitHub Actions will publish to the package registry."
