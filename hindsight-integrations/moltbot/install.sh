#!/bin/bash
set -e

echo "🚀 Installing Hindsight Memory Plugin for Moltbot..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INSTALL_DIR="$HOME/.clawdbot/extensions/hindsight-memory"

# Check Node version
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 22+"
    exit 1
fi

# Build the plugin
echo "📦 Building plugin..."
cd "$SCRIPT_DIR"
npm install
npm run build

# Deploy to Clawdbot extensions
echo "📂 Deploying to $INSTALL_DIR..."
rm -rf "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR"

# Copy files
cp -r dist package.json clawdbot.plugin.json hooks README.md "$INSTALL_DIR/"

# Install dependencies in deployed location
echo "📥 Installing dependencies..."
cd "$INSTALL_DIR"
npm install

echo ""
echo "✅ Hindsight Memory Plugin installed successfully!"
echo ""
echo "📋 Next steps:"
echo ""
echo "1. Make sure you have an OpenAI API key set:"
echo "   export OPENAI_API_KEY=\"sk-your-key-here\""
echo ""
echo "2. Enable the plugin:"
echo "   clawdbot plugins enable hindsight-memory"
echo ""
echo "3. Start Moltbot:"
echo "   clawdbot start"
echo ""
echo "On first start, uvx will automatically download hindsight-embed (no manual install needed)"
