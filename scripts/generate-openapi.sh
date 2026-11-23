#!/bin/bash
set -e

# Writes the OpenAPI spec to hindsight-docs/static/openapi.json.
# The docs site reads that file directly, so no docs build is needed here.

cd "$(dirname "$0")/../hindsight-dev"
uv run generate-openapi
