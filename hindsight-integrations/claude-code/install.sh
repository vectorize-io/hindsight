#!/bin/bash
set -e

echo "Installing Hindsight Memory Plugin for Claude Code..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found. Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python $PYTHON_VERSION"

# Check Claude Code is available
if ! command -v claude &> /dev/null; then
    echo "Warning: 'claude' command not found. Make sure Claude Code is installed."
    echo "  See: https://docs.anthropic.com/en/docs/claude-code"
fi

# Install via Claude Code plugin system
echo ""
echo "To install the plugin, run the following in Claude Code:"
echo ""
echo "  /plugin install $SCRIPT_DIR"
echo ""
echo "Or copy it manually:"
echo ""

PLUGIN_DIR="$HOME/.claude/plugins/hindsight-memory"
echo "  mkdir -p $PLUGIN_DIR"
echo "  cp -r $SCRIPT_DIR/.claude-plugin $PLUGIN_DIR/"
echo "  cp -r $SCRIPT_DIR/hooks $PLUGIN_DIR/"
echo "  cp -r $SCRIPT_DIR/scripts $PLUGIN_DIR/"
echo "  cp $SCRIPT_DIR/settings.json $PLUGIN_DIR/"

echo ""
echo "Next steps:"
echo ""
echo "1. Configure your LLM provider for memory extraction:"
echo "   # Option A: OpenAI (auto-detected)"
echo "   export OPENAI_API_KEY=\"sk-your-key\""
echo ""
echo "   # Option B: Anthropic (auto-detected)"
echo "   export ANTHROPIC_API_KEY=\"your-key\""
echo ""
echo "   # Option C: Explicit provider"
echo "   export HINDSIGHT_API_LLM_PROVIDER=openai"
echo "   export HINDSIGHT_API_LLM_API_KEY=\"sk-your-key\""
echo ""
echo "2. Or connect to an external Hindsight server:"
echo "   Edit settings.json and set hindsightApiUrl"
echo ""
echo "3. Start Claude Code — the plugin will activate automatically."
echo ""
echo "On first use with daemon mode, uvx will download hindsight-embed (no manual install needed)."
