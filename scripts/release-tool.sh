#!/bin/bash
set -e

cd "$(dirname "$0")/.."

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() { echo -e "${YELLOW}$1${NC}"; }
print_success() { echo -e "${GREEN}$1${NC}"; }
print_error() { echo -e "${RED}$1${NC}" >&2; }

VALID_TOOLS=("hindsight-agent-sdk")

usage() {
    echo "Usage: $0 <tool> <version>"
    echo "  tool     One of: ${VALID_TOOLS[*]}"
    echo "  version  Semver (e.g., 0.1.0, 1.0.0)"
    exit 1
}

if [ $# -ne 2 ]; then usage; fi

TOOL="$1"
VERSION="$2"

# Validate tool name
VALID=false
for v in "${VALID_TOOLS[@]}"; do
    if [ "$v" = "$TOOL" ]; then VALID=true; break; fi
done
if [ "$VALID" = false ]; then
    print_error "Unknown tool: $TOOL"
    print_info "Valid tools: ${VALID_TOOLS[*]}"
    exit 1
fi

# Validate version format
if ! echo "$VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$'; then
    print_error "Invalid version format: $VERSION (expected semver like 0.1.0)"
    exit 1
fi

# Resolve directory
TOOL_DIR="hindsight-tools/$TOOL"
if [ ! -d "$TOOL_DIR" ]; then
    print_error "Tool directory not found: $TOOL_DIR"
    exit 1
fi

# Check we're on main
BRANCH=$(git branch --show-current)
if [ "$BRANCH" != "main" ]; then
    print_info "Warning: not on main branch (current: $BRANCH)"
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then exit 1; fi
fi

# Check working tree is clean
if [ -n "$(git status --porcelain)" ]; then
    print_error "Working tree is not clean. Commit or stash changes first."
    exit 1
fi

# Read current version
if [ -f "$TOOL_DIR/package.json" ]; then
    CURRENT=$(grep '"version"' "$TOOL_DIR/package.json" | head -1 | sed 's/.*"version": "\(.*\)".*/\1/')
elif [ -f "$TOOL_DIR/pyproject.toml" ]; then
    CURRENT=$(grep '^version = ' "$TOOL_DIR/pyproject.toml" | sed 's/version = "\(.*\)"/\1/')
else
    CURRENT="unknown"
fi

print_info "Releasing $TOOL: $CURRENT → $VERSION"
echo

# Update version in package manifest
if [ -f "$TOOL_DIR/package.json" ]; then
    sed -i '' "s/\"version\": \"$CURRENT\"/\"version\": \"$VERSION\"/" "$TOOL_DIR/package.json"
    print_success "Updated package.json"
elif [ -f "$TOOL_DIR/pyproject.toml" ]; then
    sed -i '' "s/version = \"$CURRENT\"/version = \"$VERSION\"/" "$TOOL_DIR/pyproject.toml"
    print_success "Updated pyproject.toml"
fi

# Build
if [ -f "$TOOL_DIR/package.json" ]; then
    (cd "$TOOL_DIR" && npm run build)
    print_success "Built"
fi

# Commit, tag, push
TAG="tools/$TOOL/v$VERSION"
git add "$TOOL_DIR/"
git commit -m "release($TOOL): v$VERSION"
git tag "$TAG"
git push origin main "$TAG"

print_success "Released $TOOL v$VERSION"
print_info "Tag: $TAG"
echo
print_info "To publish to npm:"
echo "  cd $TOOL_DIR && npm publish --access public"
