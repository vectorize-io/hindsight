#!/bin/bash
set -e

cd "$(dirname "$0")/../.."

ENV_FILE=".env"
if [ ! -f "$ENV_FILE" ]; then
  echo "Error: Environment file $ENV_FILE not found at project root."
  exit 1
fi

echo "Loading environment from $ENV_FILE"
echo ""

# Export all variables from env file
set -a
source "$ENV_FILE"
set +a

uv run hindsight-api "${SERVER_ARGS[@]}"
