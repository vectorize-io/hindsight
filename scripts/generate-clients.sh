#!/usr/bin/env bash
set -e

# Script to generate Python and TypeScript clients from OpenAPI spec using openapi-generator
# Note: Rust client is auto-generated at build time via build.rs (uses progenitor)
# Usage: ./scripts/generate-clients.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CLIENTS_DIR="$PROJECT_ROOT/hindsight-clients"
OPENAPI_SPEC="$PROJECT_ROOT/hindsight-docs/static/openapi.json"

echo "=================================================="
echo "Hindsight API Client Generator"
echo "=================================================="
echo "Project root: $PROJECT_ROOT"
echo "Clients directory: $CLIENTS_DIR"
echo "OpenAPI spec: $OPENAPI_SPEC"
echo ""
echo "This script generates clients for:"
echo "  - Rust (via progenitor in build.rs)"
echo "  - Python (via openapi-generator)"
echo "  - TypeScript (via @hey-api/openapi-ts)"
echo ""

# Check if OpenAPI spec exists
if [ ! -f "$OPENAPI_SPEC" ]; then
    echo "❌ Error: OpenAPI spec not found at $OPENAPI_SPEC"
    exit 1
fi
echo "✓ OpenAPI spec found"
echo ""

# Check for Docker (we'll use Docker to run openapi-generator)
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker not found. Please install Docker"
    echo "   https://docs.docker.com/get-docker/"
    exit 1
fi
echo "✓ Docker available"
echo ""

# Generate Rust client
echo "=================================================="
echo "Generating Rust client..."
echo "=================================================="

RUST_CLIENT_DIR="$CLIENTS_DIR/rust"

# Clean old generated files
echo "Cleaning old Rust generated code..."
rm -rf "$RUST_CLIENT_DIR/target"
rm -f "$RUST_CLIENT_DIR/Cargo.lock"

# Trigger regeneration by building
echo "Regenerating Rust client (via build.rs)..."
cd "$RUST_CLIENT_DIR"
cargo clean
cargo build --release

echo "✓ Rust client generated at $RUST_CLIENT_DIR"
echo ""

# Generate Python client
echo "=================================================="
echo "Generating Python client..."
echo "=================================================="

PYTHON_CLIENT_DIR="$CLIENTS_DIR/python"

# Backup the maintained wrapper file
WRAPPER_FILE="$PYTHON_CLIENT_DIR/hindsight_client/hindsight_client.py"
WRAPPER_BACKUP="/tmp/hindsight_client_backup.py"
if [ -f "$WRAPPER_FILE" ]; then
    echo "📦 Backing up maintained wrapper: hindsight_client.py"
    cp "$WRAPPER_FILE" "$WRAPPER_BACKUP"
fi

# Backup the README.md
README_FILE="$PYTHON_CLIENT_DIR/README.md"
README_BACKUP="/tmp/hindsight_python_readme_backup.md"
if [ -f "$README_FILE" ]; then
    echo "📦 Backing up README.md"
    cp "$README_FILE" "$README_BACKUP"
fi

# Remove old generated code (but keep config and maintained files)
if [ -d "$PYTHON_CLIENT_DIR/hindsight_client_api" ]; then
    echo "Removing old generated code..."
    rm -rf "$PYTHON_CLIENT_DIR/hindsight_client_api"
fi

# Remove other generated files but keep pyproject.toml and config
for file in setup.py setup.cfg requirements.txt test-requirements.txt tox.ini git_push.sh .travis.yml .gitlab-ci.yml .gitignore README.md; do
    if [ -f "$PYTHON_CLIENT_DIR/$file" ]; then
        rm "$PYTHON_CLIENT_DIR/$file"
    fi
done

echo "Generating new client with openapi-generator..."
cd "$PYTHON_CLIENT_DIR"

# Run openapi-generator via Docker
docker run --rm \
    -v "$OPENAPI_SPEC:/local/openapi.json" \
    -v "$PYTHON_CLIENT_DIR:/local/out" \
    -v "$PYTHON_CLIENT_DIR/openapi-generator-config.yaml:/local/config.yaml" \
    openapitools/openapi-generator-cli generate \
    -i /local/openapi.json \
    -g python \
    -o /local/out \
    -c /local/config.yaml

echo "Organizing generated files..."

# The generator creates files directly, we need to ensure proper structure
# openapi-generator puts source code in agent_memory_api_client/ by default

# Restore the maintained wrapper file
if [ -f "$WRAPPER_BACKUP" ]; then
    echo "📦 Restoring maintained wrapper: hindsight_client.py"
    cp "$WRAPPER_BACKUP" "$WRAPPER_FILE"
    rm "$WRAPPER_BACKUP"
fi

# Restore the README.md
if [ -f "$README_BACKUP" ]; then
    echo "📦 Restoring README.md"
    cp "$README_BACKUP" "$README_FILE"
    rm "$README_BACKUP"
fi

# Keep our custom pyproject.toml (don't let generator overwrite it)
if [ -f "setup.py" ]; then
    echo "Note: setup.py generated but we're using pyproject.toml"
fi

# Remove the auto-generated README (we have our own)
if [ -f "$PYTHON_CLIENT_DIR/hindsight_client_api_README.md" ]; then
    echo "Removing auto-generated README..."
    rm "$PYTHON_CLIENT_DIR/hindsight_client_api_README.md"
fi

echo "✓ Python client generated at $PYTHON_CLIENT_DIR"
echo ""

# Generate TypeScript client
echo "=================================================="
echo "Generating TypeScript client..."
echo "=================================================="

TYPESCRIPT_CLIENT_DIR="$CLIENTS_DIR/typescript"

# Remove old generated client (keep package.json, tsconfig.json, tests, src/, and config)
echo "Removing old TypeScript generated code..."
rm -rf "$TYPESCRIPT_CLIENT_DIR/generated"

# Also remove legacy structure from old generator if it exists
rm -rf "$TYPESCRIPT_CLIENT_DIR/core"
rm -rf "$TYPESCRIPT_CLIENT_DIR/models"
rm -rf "$TYPESCRIPT_CLIENT_DIR/services"
rm -f "$TYPESCRIPT_CLIENT_DIR/index.ts"

# Generate new client using @hey-api/openapi-ts
echo "Generating from $OPENAPI_SPEC..."
cd "$TYPESCRIPT_CLIENT_DIR"
npx --yes @hey-api/openapi-ts

echo "✓ TypeScript client generated at $TYPESCRIPT_CLIENT_DIR"
echo ""

echo "=================================================="
echo "✅ Client generation complete!"
echo "=================================================="
echo ""
echo "Rust client:       $RUST_CLIENT_DIR"
echo "Python client:     $PYTHON_CLIENT_DIR"
echo "TypeScript client: $TYPESCRIPT_CLIENT_DIR"
echo ""
echo "⚠️  Important: The maintained wrapper hindsight_client.py and README.md were preserved"
echo ""
echo "Next steps:"
echo "  1. Review the generated clients"
echo "  2. Update package versions if needed"
echo "  3. Test the clients"
echo "  4. Run 'cargo build' in hindsight-cli to rebuild with new Rust client"
echo ""
