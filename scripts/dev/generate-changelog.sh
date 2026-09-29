#!/bin/bash
set -e

cd "$(dirname "$0")/../.."

if [ -z "$1" ]; then
  echo "Usage: $0 VERSION [--model MODEL]"
  echo ""
  echo "Generate changelog entry for a release."
  echo ""
  echo "Examples:"
  echo "  $0 1.0.5"
  echo "  $0 v1.0.5"
  echo "  $0 1.0.5 --model gpt-4o"
  exit 1
fi

if [ -z "$OPENAI_API_KEY" ]; then
  ENV_FILE=".env"
  if [ -f "$ENV_FILE" ]; then
    echo "Loading environment from $ENV_FILE"
    set -a
    source "$ENV_FILE"
    set +a
  else
    echo "Error: OPENAI_API_KEY not set and no .env file found"
    exit 1
  fi
fi

cd hindsight-dev
uv run generate-changelog "$@"
