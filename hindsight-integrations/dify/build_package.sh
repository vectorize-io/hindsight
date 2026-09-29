#!/bin/bash
# Build a .difypkg archive for upload to Dify.
# .difypkg is just a zip of the plugin root.
set -e

cd "$(dirname "$0")"

PLUGIN_NAME=$(grep "^name:" manifest.yaml | sed 's/^name:[[:space:]]*//' | tr -d '"' | tr -d "'")
VERSION=$(grep "^version:" manifest.yaml | sed 's/^version:[[:space:]]*//' | tr -d '"' | tr -d "'")
OUTPUT_FILE="${PLUGIN_NAME}-${VERSION}.difypkg"
TEMP_DIR=$(mktemp -d)

echo "Building ${OUTPUT_FILE}..."

# Files to ship in the package
for path in manifest.yaml main.py requirements.txt PRIVACY.md README.md LICENSE _assets provider tools; do
  if [ -e "$path" ]; then
    cp -r "$path" "$TEMP_DIR/"
  fi
done

# Strip caches and OS junk
find "$TEMP_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$TEMP_DIR" -type f \( -name "*.pyc" -o -name ".DS_Store" \) -delete 2>/dev/null || true

rm -f "$OUTPUT_FILE"
( cd "$TEMP_DIR" && zip -r -D -q "$OLDPWD/$OUTPUT_FILE" . )
rm -rf "$TEMP_DIR"

echo "Created $(pwd)/${OUTPUT_FILE} ($(du -h "$OUTPUT_FILE" | cut -f1))"
