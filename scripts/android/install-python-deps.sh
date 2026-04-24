#!/data/data/com.termux/files/usr/bin/bash
# Install Python dependencies for Hindsight on Android/Termux.
#
# IMPORTANT: Some packages (pydantic-core, jiter, tiktoken, greenlet, uvloop, etc.)
# require Rust/C compilation which can take 30-60+ minutes on a phone CPU.
# Make sure:
#   - termux-wake-lock is active (prevents Android from killing the process)
#   - Phone is plugged in (compilation is CPU-intensive and drains battery)
#   - You have at least 3GB free disk space
#
# Usage: bash scripts/android/install-python-deps.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
API_DIR="$REPO_ROOT/hindsight-api-slim"

echo "=== Installing Hindsight Python Dependencies ==="
echo "This may take 30-60+ minutes due to Rust/C compilation on ARM."
echo ""

# Ensure wake lock
termux-wake-lock 2>/dev/null || true

# Create virtualenv if not exists
if [ ! -d "$API_DIR/.venv" ]; then
    echo ">>> Creating virtualenv..."
    pip install --break-system-packages virtualenv 2>/dev/null || pip install virtualenv
    virtualenv "$API_DIR/.venv"
fi

source "$API_DIR/.venv/bin/activate"

# Install packages in groups to isolate failures.
# Group 1: Pure Python wheels (fast, no compilation)
echo ""
echo ">>> [1/4] Installing pure Python packages..."
pip install \
    python-dotenv \
    rich \
    distro \
    sniffio \
    idna \
    certifi \
    click \
    Mako \
    MarkupSafe \
    python-dateutil \
    six \
    wsproto \
    h11 \
    2>&1 | tail -3

# Group 2: Packages with C extensions (moderate compile time)
echo ""
echo ">>> [2/4] Installing C-extension packages (5-10 min)..."
for pkg in asyncpg psycopg2-binary greenlet uvloop httptools; do
    echo "  Installing $pkg..."
    pip install "$pkg" 2>&1 | tail -2
    if [ $? -ne 0 ]; then
        echo "  WARNING: $pkg failed to install, will try to continue"
    fi
done

# Group 3: Rust-dependent packages (slowest - 20-40 min each)
echo ""
echo ">>> [3/4] Installing Rust-dependent packages (this is the slow part)..."
echo "  Phone CPU will be at 100% — this is normal."
for pkg in pydantic-core jiter tiktoken; do
    echo "  Installing $pkg (may take 10-30 min)..."
    pip install "$pkg" 2>&1 | tail -3
    if [ $? -ne 0 ]; then
        echo "  WARNING: $pkg failed. This may cause issues."
        echo "  You can retry individually: pip install $pkg"
    fi
done

# Group 4: Higher-level packages (mostly pure Python, depend on groups above)
echo ""
echo ">>> [4/4] Installing framework packages..."
pip install \
    pydantic \
    openai \
    httpx \
    httpcore \
    anyio \
    typing-extensions \
    annotated-types \
    fastapi \
    uvicorn \
    starlette \
    sqlalchemy \
    alembic \
    pgvector \
    typer \
    2>&1 | tail -3

echo ""
echo ">>> Verifying installation..."
python -c "
import asyncpg
import fastapi
import sqlalchemy
import pydantic
print('Core packages: OK')
" 2>&1 || echo "WARNING: Some core packages missing"

python -c "
import openai
print('OpenAI SDK: OK')
" 2>&1 || echo "WARNING: OpenAI SDK not working (jiter may be missing)"

echo ""
echo "=== Python dependency installation complete ==="
echo "Installed packages:"
pip list 2>/dev/null | wc -l
echo ""
echo "Next: bash scripts/android/start-api.sh"
