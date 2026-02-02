#!/bin/bash
set -e

# Unified docs versioning script for Docusaurus
# Automatically handles both patch releases (sync) and minor/major releases (create new version)
#
# Usage: ./scripts/update-docs-version.sh <version>
# Examples:
#   ./scripts/update-docs-version.sh 0.4.2  # Patch: syncs docs/ to existing version-0.4/
#   ./scripts/update-docs-version.sh 0.5.0  # Minor: creates new version-0.5/ snapshot

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DOCS_DIR="$ROOT_DIR/hindsight-docs"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

if [ -z "$1" ]; then
    echo "Usage: $0 <version>"
    echo ""
    echo "Examples:"
    echo "  $0 0.4.2  # Patch release: syncs docs/ to version-0.4/"
    echo "  $0 0.5.0  # Minor release: creates new version-0.5/"
    exit 1
fi

VERSION="$1"

# Validate version format (semantic versioning)
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Version must be in X.Y.Z format (e.g., 0.4.2 or 0.5.0)"
    exit 1
fi

# Extract major.minor and patch version
MAJOR_MINOR=$(echo "$VERSION" | sed -E 's/^([0-9]+\.[0-9]+)\.[0-9]+$/\1/')
PATCH_VERSION=$(echo "$VERSION" | sed -E 's/^[0-9]+\.[0-9]+\.([0-9]+)$/\1/')

SOURCE_DIR="$DOCS_DIR/docs"
TARGET_DIR="$DOCS_DIR/versioned_docs/version-${MAJOR_MINOR}"
VERSIONS_FILE="$DOCS_DIR/versions.json"

# Determine action based on patch version
if [ "$PATCH_VERSION" != "0" ]; then
    #
    # PATCH RELEASE: Sync docs to existing version
    #
    print_info "Detected PATCH release ($VERSION)"
    print_info "Syncing docs/ → versioned_docs/version-${MAJOR_MINOR}/"

    if [ ! -d "$TARGET_DIR" ]; then
        echo "Error: Target version directory does not exist: $TARGET_DIR"
        echo ""
        echo "Available versions:"
        [ -f "$VERSIONS_FILE" ] && cat "$VERSIONS_FILE" || echo "  (No versions.json found)"
        exit 1
    fi

    # Use rsync to sync, preserving structure and deleting removed files
    rsync -av --delete \
        --exclude='*.swp' \
        --exclude='.DS_Store' \
        "$SOURCE_DIR/" "$TARGET_DIR/"

    # Sync sidebar configuration
    print_info "Syncing sidebars.ts → versioned_sidebars/version-${MAJOR_MINOR}-sidebars.json"

    cd "$DOCS_DIR"
    # Use Node.js to convert sidebars.ts to JSON and update versioned sidebar
    node -e "
    const sidebars = require('./sidebars.ts').default;
    const fs = require('fs');
    const targetFile = './versioned_sidebars/version-${MAJOR_MINOR}-sidebars.json';
    fs.writeFileSync(targetFile, JSON.stringify(sidebars, null, 2) + '\n');
    console.log('Updated: ' + targetFile);
    "

    echo ""
    print_info "✓ Synced docs/ to version-${MAJOR_MINOR}"
    print_info "✓ Synced sidebars to version-${MAJOR_MINOR}-sidebars.json"
    print_info "✓ Files updated in: $TARGET_DIR"

else
    #
    # MINOR/MAJOR RELEASE: Create new version snapshot
    #
    print_info "Detected MINOR/MAJOR release ($VERSION)"
    print_info "Creating new docs version: version-${MAJOR_MINOR}"

    # Check if version already exists
    if [ -d "$TARGET_DIR" ]; then
        echo "Error: Version $MAJOR_MINOR already exists in $TARGET_DIR"
        echo "If you want to update it, use a patch version (e.g., ${MAJOR_MINOR}.1)"
        exit 1
    fi

    # Create the version snapshot using Docusaurus
    cd "$DOCS_DIR"
    npx docusaurus docs:version "$MAJOR_MINOR"

    echo ""
    print_info "✓ Created docs version-${MAJOR_MINOR}"
    print_info "Files created:"
    print_info "  - versioned_docs/version-${MAJOR_MINOR}/"
    print_info "  - versioned_sidebars/version-${MAJOR_MINOR}-sidebars.json"
    print_info "  - versions.json (updated)"
    echo ""
    print_warn "IMPORTANT: Future docs changes will go to docs/ (next version)"
    print_warn "           To update ${MAJOR_MINOR} docs, use patch releases (e.g., ${MAJOR_MINOR}.1)"
fi

echo ""
print_info "Next steps:"
echo "  1. Review changes: git diff $DOCS_DIR"
echo "  2. Test build: cd hindsight-docs && npm run build"
echo "  3. Changes will be committed automatically by release.sh"
