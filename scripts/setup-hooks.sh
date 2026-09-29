#!/bin/bash
# Setup git hooks for the repository

set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"

echo "Setting up git hooks..."
git config core.hooksPath "$REPO_ROOT/.githooks"
echo "Git hooks configured to use .githooks directory"
echo "Done!"
