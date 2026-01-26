#!/bin/bash
set -e

# Bump documentation version for Docusaurus
# Usage: ./scripts/bump-docs-version.sh <version>
# Example: ./scripts/bump-docs-version.sh 0.4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DOCS_DIR="$ROOT_DIR/hindsight-docs"

if [ -z "$1" ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 0.4"
    exit 1
fi

VERSION="$1"

# Validate version format (minor version only: X.Y)
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Version must be in format X.Y (e.g., 0.4)"
    exit 1
fi

echo "Creating docs version $VERSION..."

# Create the version snapshot
cd "$DOCS_DIR"
npx docusaurus docs:version "$VERSION"

echo ""
echo "Done! Version $VERSION created."
echo ""
echo "Files created/modified:"
echo "  - versioned_docs/version-$VERSION/"
echo "  - versioned_sidebars/version-$VERSION-sidebars.json"
echo "  - versions.json (automatically read by docusaurus.config.ts)"
echo ""
echo "Next steps:"
echo "  1. Review the changes"
echo "  2. Test with: cd hindsight-docs && npm run build"
echo "  3. Commit the versioned docs"
