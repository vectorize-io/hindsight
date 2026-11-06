#!/bin/bash
# Setup git hooks for the repository

set -e

# A relative core.hooksPath resolves against each worktree's own root. The
# setting lives in the shared .git/config, so an absolute path would point
# every worktree at whichever one ran this last.
current="$(git config --get core.hooksPath || true)"

# Unset, already relative, or the absolute form older versions of this script
# wrote: (re)write it. Anything else is the developer's own choice: keep it.
if [ -z "$current" ] || [ "$current" = ".githooks" ] || [[ "$current" == */.githooks ]]; then
  git config core.hooksPath .githooks
  echo "Git hooks configured to use .githooks directory"
else
  echo "core.hooksPath is set to '$current'; leaving it alone"
fi
