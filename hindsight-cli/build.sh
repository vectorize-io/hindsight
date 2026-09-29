#!/bin/bash
set -e

# Build script for hindsight-cli
# This script builds optimized binaries for multiple platforms

# Source cargo environment if it exists
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

# Add cargo to PATH if not already there
export PATH="$HOME/.cargo/bin:$PATH"

# Check if cargo is available
if ! command -v cargo &> /dev/null; then
    echo "Error: Cargo not found. Please install Rust first:"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

echo "Building Hindsight CLI for multiple platforms..."

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Create dist directory if it doesn't exist
mkdir -p dist

# Get version from Cargo.toml
VERSION=$(grep '^version' Cargo.toml | head -1 | cut -d'"' -f2)
echo "Version: $VERSION"

# Function to build for a target
build_target() {
    local target=$1
    local output_name=$2

    echo ""
    echo "Building for $target..."

    # Check if target is installed, install if not
    if ! rustup target list | grep -q "$target (installed)"; then
        echo "Installing target $target..."
        rustup target add "$target"
    fi

    # Build
    cargo build --release --target "$target"

    # Copy to dist
    if [[ "$target" == *"windows"* ]]; then
        cp "target/$target/release/hindsight.exe" "dist/$output_name.exe"
        echo "Created: dist/$output_name.exe"
    else
        cp "target/$target/release/hindsight" "dist/$output_name"
        chmod +x "dist/$output_name"
        echo "Created: dist/$output_name"
    fi
}

# Detect current platform
OS=$(uname -s)
ARCH=$(uname -m)

echo "Detected platform: $OS $ARCH"

# Build for current platform
case "$OS" in
    Darwin)
        if [[ "$ARCH" == "arm64" ]]; then
            echo "Building for macOS ARM64 (Apple Silicon)..."
            build_target "aarch64-apple-darwin" "hindsight-macos-arm64"
        else
            echo "Building for macOS x86_64 (Intel)..."
            build_target "x86_64-apple-darwin" "hindsight-macos-x86_64"
        fi
        ;;
    Linux)
        if [[ "$ARCH" == "x86_64" ]]; then
            echo "Building for Linux x86_64..."
            build_target "x86_64-unknown-linux-gnu" "hindsight-linux-x86_64"
        elif [[ "$ARCH" == "aarch64" ]]; then
            echo "Building for Linux ARM64..."
            build_target "aarch64-unknown-linux-gnu" "hindsight-linux-arm64"
        fi
        ;;
    *)
        echo "Unsupported OS: $OS"
        exit 1
        ;;
esac

echo ""
echo "Build complete! Binaries are in the dist/ directory:"
ls -lh dist/

echo ""
echo "To build for other platforms, run:"
echo "  ./build.sh --all"
