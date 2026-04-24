#!/bin/bash
# Cross-compile Rust Python wheels for Android ARM64 using the Android NDK.
# Run this on a Mac build machine, then transfer wheels to the device.
#
# Prerequisites:
#   - Android NDK: brew install --cask android-ndk
#   - Rust Android target: rustup target add aarch64-linux-android
#   - Python 3.13 (matching device): pyenv install 3.13.2 OR uv python install 3.13
#   - maturin: pip install maturin
#
# Usage:
#   bash scripts/android/build-wheels.sh
#   # Then copy wheels to device and install:
#   scp -P 8022 dist/android-arm64/*.whl user@device:~/wheels/
#   ssh -p 8022 user@device 'cd ~/hindsight/hindsight-api-slim && source .venv/bin/activate && pip install ~/wheels/*.whl'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WHEEL_DIR="$REPO_ROOT/dist/android-arm64"
mkdir -p "$WHEEL_DIR"

# ── Find Android NDK ─────────────────────────────────────────────
ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-/opt/homebrew/share/android-ndk}"
if [ ! -d "$ANDROID_NDK_HOME" ]; then
    echo "ERROR: Android NDK not found at $ANDROID_NDK_HOME"
    echo "Install: brew install --cask android-ndk"
    exit 1
fi
TOOLCHAIN="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64"
CC="$TOOLCHAIN/bin/aarch64-linux-android24-clang"

# ── Find Python 3.13 ─────────────────────────────────────────────
PYTHON313=""
for candidate in \
    "$(uv python find 3.13 2>/dev/null || true)" \
    "$HOME/.pyenv/versions/3.13.2/bin/python3.13" \
    "$(which python3.13 2>/dev/null || true)"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
        PYTHON313="$candidate"
        break
    fi
done
if [ -z "$PYTHON313" ]; then
    echo "ERROR: Python 3.13 not found. Install via: pyenv install 3.13.2"
    exit 1
fi
echo "Using Python: $PYTHON313"

# ── Check Rust target ─────────────────────────────────────────────
if ! rustup target list --installed | grep -q aarch64-linux-android; then
    echo ">>> Adding Rust Android target..."
    rustup target add aarch64-linux-android
fi

# ── Check maturin ─────────────────────────────────────────────────
if ! command -v maturin &>/dev/null; then
    echo ">>> Installing maturin..."
    pip install maturin
fi

# ── Set cross-compile environment ─────────────────────────────────
export CC_aarch64_linux_android="$CC"
export CXX_aarch64_linux_android="$TOOLCHAIN/bin/aarch64-linux-android24-clang++"
export AR_aarch64_linux_android="$TOOLCHAIN/bin/llvm-ar"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$CC"
export PYO3_CROSS_PYTHON_VERSION=3.13
export PYO3_CROSS=1

# Create stub libpython for linking (actual resolution happens at runtime on device)
CROSS_LIB_DIR="$(mktemp -d)"
cat > "$CROSS_LIB_DIR/_sysconfigdata__android_arm64_v8a.py" << 'SYSCONFIG'
build_time_vars = {
    'SOABI': 'cpython-313-aarch64-linux-android',
    'EXT_SUFFIX': '.cpython-313-aarch64-linux-android.so',
    'LDLIBRARY': 'libpython3.13.so',
    'LIBDIR': '/data/data/com.termux/files/usr/lib',
    'Py_ENABLE_SHARED': 1,
    'MULTIARCH': 'aarch64-linux-android',
    'VERSION': '3.13',
    'ABIFLAGS': '',
}
SYSCONFIG
echo "void Py_Initialize(void) {}" > "$CROSS_LIB_DIR/stub.c"
$CC -shared -o "$CROSS_LIB_DIR/libpython3.13.so" "$CROSS_LIB_DIR/stub.c"

export PYO3_CROSS_LIB_DIR="$CROSS_LIB_DIR"
export _PYTHON_SYSCONFIGDATA_NAME=_sysconfigdata__android_arm64_v8a
export PYTHONPATH="$CROSS_LIB_DIR"

# ── Build wheels ──────────────────────────────────────────────────
PACKAGES=(
    "pydantic-core"
    "jiter"
    "tiktoken"
)

WORK_DIR="$(mktemp -d)"

echo "=== Building Android ARM64 wheels ==="
for pkg in "${PACKAGES[@]}"; do
    echo ""
    echo ">>> Building $pkg..."
    pkg_dir="$WORK_DIR/$pkg"
    mkdir -p "$pkg_dir"

    # Download source
    pip download --no-binary :all: --no-deps "$pkg" -d "$pkg_dir" 2>&1 | tail -1

    # Extract
    cd "$pkg_dir"
    tar xzf *.tar.gz
    src_dir=$(find . -maxdepth 1 -type d -name "${pkg//-/_}*" -o -name "${pkg}*" | grep -v '.tar.gz' | head -1)

    # Find the Cargo.toml (some packages nest it)
    cargo_dir=$(find "$src_dir" -name Cargo.toml -maxdepth 3 -exec dirname {} \; | head -1)
    cd "$cargo_dir"

    # Build
    maturin build --release --target aarch64-linux-android -i "$PYTHON313" -o "$WHEEL_DIR" 2>&1 | tail -3

    echo "  Built: $(ls "$WHEEL_DIR"/*${pkg//-/_}* 2>/dev/null | tail -1)"
done

# ── Cleanup ───────────────────────────────────────────────────────
rm -rf "$WORK_DIR" "$CROSS_LIB_DIR"

echo ""
echo "=== All wheels built ==="
ls -lh "$WHEEL_DIR"/*.whl
echo ""
echo "Next: copy to device and install"
echo "  scp -P 8022 $WHEEL_DIR/*.whl user@device:~/wheels/"
echo "  ssh -p 8022 user@device 'pip install ~/wheels/*.whl'"
