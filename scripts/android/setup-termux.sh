#!/data/data/com.termux/files/usr/bin/bash
# Hindsight Android/Termux setup script
# Prerequisites: Install Termux from F-Droid (NOT Play Store)
#
# Usage:
#   1. Open Termux on your Android device
#   2. pkg install git
#   3. git clone <repo-url> ~/hindsight
#   4. cd ~/hindsight && bash scripts/android/setup-termux.sh
#
# This script installs all system dependencies and sets up PostgreSQL + pgvector.
# After running this, use scripts/android/install-python-deps.sh for Python packages.

set -euo pipefail

echo "=== Hindsight Android/Termux Setup ==="
echo "Device: $(uname -m) / $(uname -s)"
echo "Available RAM: $(free -h | awk '/Mem:/{print $2}')"
echo "Available disk: $(df -h /data 2>/dev/null | awk 'NR==2{print $4}' || echo 'unknown')"
echo ""

# Prevent Android from killing Termux in background
echo ">>> Acquiring wake lock..."
termux-wake-lock 2>/dev/null || true

# ── System packages ──────────────────────────────────────────────
echo ">>> Installing system packages..."
pkg update -y
pkg upgrade -y
pkg install -y \
    python \
    postgresql \
    git \
    make \
    clang \
    pkg-config \
    rust \
    openssh

# ── PostgreSQL setup ─────────────────────────────────────────────
PG_DATA="$HOME/pg-data"
if [ ! -d "$PG_DATA" ]; then
    echo ">>> Initializing PostgreSQL..."
    initdb -D "$PG_DATA"
else
    echo ">>> PostgreSQL data directory already exists, skipping initdb"
fi

# Start PostgreSQL if not running
if ! pg_isready -q 2>/dev/null; then
    echo ">>> Starting PostgreSQL..."
    pg_ctl -D "$PG_DATA" -l "$HOME/pg.log" start
    sleep 2
fi

# Create database
if ! psql -lqt 2>/dev/null | cut -d \| -f 1 | grep -qw hindsight; then
    echo ">>> Creating hindsight database..."
    createdb hindsight
else
    echo ">>> Database 'hindsight' already exists"
fi

# ── pgvector extension ───────────────────────────────────────────
if ! psql -d hindsight -c "SELECT extversion FROM pg_extension WHERE extname='vector'" 2>/dev/null | grep -q .; then
    echo ">>> Building pgvector from source..."
    PGVECTOR_DIR="$HOME/pgvector"
    if [ ! -d "$PGVECTOR_DIR" ]; then
        git clone --depth 1 https://github.com/pgvector/pgvector.git "$PGVECTOR_DIR"
    fi
    cd "$PGVECTOR_DIR"
    git checkout master
    git pull
    make clean
    PG_LDFLAGS="-lm" make
    make install
    psql -d hindsight -c "CREATE EXTENSION IF NOT EXISTS vector;"
    cd -
    echo ">>> pgvector installed and enabled"
else
    echo ">>> pgvector already installed"
fi

# Verify pgvector works
echo ">>> Verifying pgvector..."
psql -d hindsight -c "SELECT '[1,2,3]'::vector <-> '[4,5,6]'::vector AS test_distance;" || {
    echo "ERROR: pgvector verification failed"
    exit 1
}

# ── SSH server (for remote access) ───────────────────────────────
echo ">>> Setting up SSH server..."
echo "  To enable remote access:"
echo "    1. Run: passwd (set a password)"
echo "    2. Run: sshd"
echo "    3. Connect from Mac: ssh -p 8022 \$(whoami)@<device-ip>"

# ── Summary ──────────────────────────────────────────────────────
echo ""
echo "=== Setup Complete ==="
echo "  PostgreSQL: running (data at $PG_DATA)"
echo "  pgvector: installed"
echo "  Database: hindsight"
echo "  Database URL: postgresql://$(whoami)@localhost/hindsight"
echo ""
echo "Next steps:"
echo "  1. Run: bash scripts/android/install-python-deps.sh"
echo "  2. Run: bash scripts/android/start-api.sh"
