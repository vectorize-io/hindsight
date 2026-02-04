#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if version is provided
if [ -z "$1" ]; then
    print_error "Usage: $0 <version>"
    print_info "Example: $0 0.2.0"
    exit 1
fi

VERSION=$1

# Validate version format (semantic versioning)
if ! [[ $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    print_error "Invalid version format. Please use semantic versioning (e.g., 0.2.0)"
    exit 1
fi

print_info "Starting release process for version $VERSION"

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
if git rev-parse "v$VERSION" >/dev/null 2>&1; then
    print_error "Tag v$VERSION already exists"
    exit 1
fi

print_info "Updating version in all components..."

# Update Python packages
PYTHON_PACKAGES=("hindsight-api" "hindsight-dev" "hindsight" "hindsight-integrations/litellm" "hindsight-embed")
for package in "${PYTHON_PACKAGES[@]}"; do
    PYPROJECT_FILE="$package/pyproject.toml"
    if [ -f "$PYPROJECT_FILE" ]; then
        print_info "Updating $PYPROJECT_FILE"
        sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" "$PYPROJECT_FILE"
        rm "${PYPROJECT_FILE}.bak"
    else
        print_warn "File $PYPROJECT_FILE not found, skipping"
    fi
done

# Update __version__ in Python __init__.py files
PYTHON_INIT_FILES=(
    "hindsight-api/hindsight_api/__init__.py"
    "hindsight-embed/hindsight_embed/__init__.py"
    "hindsight-clients/python/hindsight_client_api/__init__.py"
)
for init_file in "${PYTHON_INIT_FILES[@]}"; do
    if [ -f "$init_file" ]; then
        print_info "Updating __version__ in $init_file"
        sed -i.bak "s/^__version__ = \".*\"/__version__ = \"$VERSION\"/" "$init_file"
        rm "${init_file}.bak"
    else
        print_warn "File $init_file not found, skipping"
    fi
done

# Update Rust CLI
CARGO_FILE="hindsight-cli/Cargo.toml"
if [ -f "$CARGO_FILE" ]; then
    print_info "Updating $CARGO_FILE"
    sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" "$CARGO_FILE"
    rm "${CARGO_FILE}.bak"
else
    print_warn "File $CARGO_FILE not found, skipping"
fi

# Update Helm chart
HELM_CHART_FILE="helm/hindsight/Chart.yaml"
if [ -f "$HELM_CHART_FILE" ]; then
    print_info "Updating $HELM_CHART_FILE"
    sed -i.bak "s/^version: .*/version: $VERSION/" "$HELM_CHART_FILE"
    sed -i.bak "s/^appVersion: .*/appVersion: \"$VERSION\"/" "$HELM_CHART_FILE"
    rm "${HELM_CHART_FILE}.bak"
else
    print_warn "File $HELM_CHART_FILE not found, skipping"
fi

# Update Control Plane package.json
CONTROL_PLANE_PKG="hindsight-control-plane/package.json"
if [ -f "$CONTROL_PLANE_PKG" ]; then
    print_info "Updating $CONTROL_PLANE_PKG"
    sed -i.bak "s/\"version\": \".*\"/\"version\": \"$VERSION\"/" "$CONTROL_PLANE_PKG"
    rm "${CONTROL_PLANE_PKG}.bak"
else
    print_warn "File $CONTROL_PLANE_PKG not found, skipping"
fi

# Update Python API client
PYTHON_CLIENT_PKG="hindsight-clients/python/pyproject.toml"
if [ -f "$PYTHON_CLIENT_PKG" ]; then
    print_info "Updating $PYTHON_CLIENT_PKG"
    sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" "$PYTHON_CLIENT_PKG"
    rm "${PYTHON_CLIENT_PKG}.bak"
else
    print_warn "File $PYTHON_CLIENT_PKG not found, skipping"
fi

# Update TypeScript API client
TYPESCRIPT_CLIENT_PKG="hindsight-clients/typescript/package.json"
if [ -f "$TYPESCRIPT_CLIENT_PKG" ]; then
    print_info "Updating $TYPESCRIPT_CLIENT_PKG"
    sed -i.bak "s/\"version\": \".*\"/\"version\": \"$VERSION\"/" "$TYPESCRIPT_CLIENT_PKG"
    rm "${TYPESCRIPT_CLIENT_PKG}.bak"
else
    print_warn "File $TYPESCRIPT_CLIENT_PKG not found, skipping"
fi

# Update OpenClaw integration
OPENCLAW_PKG="hindsight-integrations/openclaw/package.json"
if [ -f "$OPENCLAW_PKG" ]; then
    print_info "Updating $OPENCLAW_PKG"
    sed -i.bak "s/\"version\": \".*\"/\"version\": \"$VERSION\"/" "$OPENCLAW_PKG"
    rm "${OPENCLAW_PKG}.bak"
else
    print_warn "File $OPENCLAW_PKG not found, skipping"
fi

# Update AI SDK integration
AI_SDK_PKG="hindsight-integrations/ai-sdk/package.json"
if [ -f "$AI_SDK_PKG" ]; then
    print_info "Updating $AI_SDK_PKG"
    sed -i.bak "s/\"version\": \".*\"/\"version\": \"$VERSION\"/" "$AI_SDK_PKG"
    rm "${AI_SDK_PKG}.bak"
else
    print_warn "File $AI_SDK_PKG not found, skipping"
fi

# Update documentation version (creates new version or syncs to existing)
print_info "Updating documentation for version $VERSION..."
if [ -f "scripts/update-docs-version.sh" ]; then
    ./scripts/update-docs-version.sh "$VERSION" 2>&1 | grep -E "✓|IMPORTANT|Error" || true
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        print_info "✓ Documentation updated"
    else
        print_warn "Failed to update documentation, but continuing..."
    fi
else
    print_warn "update-docs-version.sh not found, skipping docs update"
fi

# Regenerate OpenAPI spec and clients with new version
print_info "Regenerating OpenAPI spec and client SDKs..."
if ./scripts/generate-openapi.sh && ./scripts/generate-clients.sh; then
    print_info "✓ OpenAPI spec and clients regenerated"
else
    print_error "Failed to regenerate clients"
    print_warn "You may need to fix this manually before committing"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_error "Release cancelled. Rolling back changes..."
        git checkout .
        exit 1
    fi
fi

# Commit changes
print_info "Committing version changes..."
git add -A

# Extract major.minor and patch for commit message
MAJOR_MINOR=$(echo "$VERSION" | sed -E 's/^([0-9]+\.[0-9]+)\.[0-9]+$/\1/')
PATCH_VERSION=$(echo "$VERSION" | sed -E 's/^[0-9]+\.[0-9]+\.([0-9]+)$/\1/')

# Build commit message
COMMIT_MSG="Release v$VERSION

- Update version to $VERSION in all components
- Regenerate OpenAPI spec and client SDKs
- Python packages: hindsight-api, hindsight-dev, hindsight-all, hindsight-litellm, hindsight-embed
- Python client: hindsight-clients/python
- TypeScript client: hindsight-clients/typescript
- Rust CLI: hindsight-cli
- Control Plane: hindsight-control-plane
- OpenClaw integration: hindsight-integrations/openclaw
- AI SDK integration: hindsight-integrations/ai-sdk
- Helm chart"

# Add docs update note
if [ "$PATCH_VERSION" != "0" ]; then
    COMMIT_MSG="$COMMIT_MSG
- Sync documentation to version-$MAJOR_MINOR"
else
    COMMIT_MSG="$COMMIT_MSG
- Create documentation version-$MAJOR_MINOR"
fi

git commit --no-verify -m "$COMMIT_MSG"

# Create tag
print_info "Creating tag v$VERSION..."
git tag -a "v$VERSION" -m "Release v$VERSION"

# Push changes
print_info "Pushing changes and tag to remote..."
git push origin "$CURRENT_BRANCH"
git push origin "v$VERSION"

print_info "✅ Release v$VERSION completed successfully!"
print_info "GitHub Actions will now build the release artifacts."
print_info "Tag: v$VERSION"
