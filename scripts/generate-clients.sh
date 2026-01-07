#!/usr/bin/env bash
set -e

# Script to generate Python and TypeScript clients from OpenAPI spec using openapi-generator
# Note: Rust client is auto-generated at build time via build.rs (uses progenitor)
# Usage: ./scripts/generate-clients.sh

# Pin openapi-generator version for reproducible builds across local and CI
OPENAPI_GENERATOR_VERSION="v7.10.0"

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
echo "✓ Using openapi-generator ${OPENAPI_GENERATOR_VERSION}"
echo ""

# Generate Rust client
echo "=================================================="
echo "Generating Rust client..."
echo "=================================================="

RUST_CLIENT_DIR="$CLIENTS_DIR/rust"

# Clean old generated files (keep Cargo.lock for reproducible builds)
echo "Cleaning old Rust generated code..."
rm -rf "$RUST_CLIENT_DIR/target"

# Trigger regeneration by building
# Use --locked to ensure reproducible builds from committed Cargo.lock
echo "Regenerating Rust client (via build.rs)..."
cd "$RUST_CLIENT_DIR"
cargo clean
cargo build --release --locked

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

# Run openapi-generator via Docker (pinned version for reproducibility)
# Use --user to match current user's UID/GID so generated files are writable
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "$OPENAPI_SPEC:/local/openapi.json" \
    -v "$PYTHON_CLIENT_DIR:/local/out" \
    -v "$PYTHON_CLIENT_DIR/openapi-generator-config.yaml:/local/config.yaml" \
    "openapitools/openapi-generator-cli:${OPENAPI_GENERATOR_VERSION}" generate \
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

# Patch rest.py to defer aiohttp initialization (fixes "no running event loop" error)
# The generated code creates aiohttp.TCPConnector in __init__ which requires a running event loop.
# We patch it to defer initialization until the first request (which runs in async context).
echo "Patching rest.py for deferred aiohttp initialization..."
REST_FILE="$PYTHON_CLIENT_DIR/hindsight_client_api/rest.py"
if [ -f "$REST_FILE" ]; then
    cd "$PROJECT_ROOT"
    python3 << PATCH_SCRIPT
import re

rest_file = "$PYTHON_CLIENT_DIR/hindsight_client_api/rest.py"

with open(rest_file, 'r') as f:
    content = f.read()

# Replace the __init__ method to defer initialization
old_init = '''class RESTClientObject:

    def __init__(self, configuration) -> None:

        # maxsize is number of requests to host that are allowed in parallel
        maxsize = configuration.connection_pool_maxsize

        ssl_context = ssl.create_default_context(
            cafile=configuration.ssl_ca_cert
        )
        if configuration.cert_file:
            ssl_context.load_cert_chain(
                configuration.cert_file, keyfile=configuration.key_file
            )

        if not configuration.verify_ssl:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        connector = aiohttp.TCPConnector(
            limit=maxsize,
            ssl=ssl_context
        )

        self.proxy = configuration.proxy
        self.proxy_headers = configuration.proxy_headers

        # https pool manager
        self.pool_manager = aiohttp.ClientSession(
            connector=connector,
            trust_env=True
        )

        retries = configuration.retries
        self.retry_client: Optional[aiohttp_retry.RetryClient]
        if retries is not None:
            self.retry_client = aiohttp_retry.RetryClient(
                client_session=self.pool_manager,
                retry_options=aiohttp_retry.ExponentialRetry(
                    attempts=retries,
                    factor=2.0,
                    start_timeout=0.1,
                    max_timeout=120.0
                )
            )
        else:
            self.retry_client = None'''

new_init = '''class RESTClientObject:

    def __init__(self, configuration) -> None:
        # Store configuration for deferred initialization
        # aiohttp.TCPConnector requires a running event loop, so we defer
        # creation until the first request (which runs in async context)
        self._configuration = configuration
        self._pool_manager: Optional[aiohttp.ClientSession] = None
        self._retry_client: Optional[aiohttp_retry.RetryClient] = None

        self.proxy = configuration.proxy
        self.proxy_headers = configuration.proxy_headers

    def _ensure_session(self) -> None:
        """Create aiohttp session lazily (must be called from async context)."""
        if self._pool_manager is not None:
            return

        configuration = self._configuration
        maxsize = configuration.connection_pool_maxsize

        ssl_context = ssl.create_default_context(
            cafile=configuration.ssl_ca_cert
        )
        if configuration.cert_file:
            ssl_context.load_cert_chain(
                configuration.cert_file, keyfile=configuration.key_file
            )

        if not configuration.verify_ssl:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        connector = aiohttp.TCPConnector(
            limit=maxsize,
            ssl=ssl_context
        )

        self._pool_manager = aiohttp.ClientSession(
            connector=connector,
            trust_env=True
        )

        retries = configuration.retries
        if retries is not None:
            self._retry_client = aiohttp_retry.RetryClient(
                client_session=self._pool_manager,
                retry_options=aiohttp_retry.ExponentialRetry(
                    attempts=retries,
                    factor=2.0,
                    start_timeout=0.1,
                    max_timeout=120.0
                )
            )

    @property
    def pool_manager(self) -> aiohttp.ClientSession:
        """Get the pool manager, initializing if needed."""
        self._ensure_session()
        return self._pool_manager

    @property
    def retry_client(self) -> Optional[aiohttp_retry.RetryClient]:
        """Get the retry client, initializing if needed."""
        self._ensure_session()
        return self._retry_client'''

if old_init in content:
    content = content.replace(old_init, new_init)

    # Also update the close method to handle None pool_manager
    old_close = '''    async def close(self):
        await self.pool_manager.close()
        if self.retry_client is not None:
            await self.retry_client.close()'''

    new_close = '''    async def close(self):
        if self._pool_manager is not None:
            await self._pool_manager.close()
        if self._retry_client is not None:
            await self._retry_client.close()'''

    content = content.replace(old_close, new_close)

    with open(rest_file, 'w') as f:
        f.write(content)
    print("  ✓ rest.py patched successfully")
else:
    print("  ⚠ Could not find expected pattern in rest.py - skipping patch")
PATCH_SCRIPT
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
# Use npm run generate to use the locally installed version (pinned in package.json)
# instead of npx --yes which would fetch the latest version
echo "Generating from $OPENAPI_SPEC..."
cd "$TYPESCRIPT_CLIENT_DIR"
npm run generate

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
