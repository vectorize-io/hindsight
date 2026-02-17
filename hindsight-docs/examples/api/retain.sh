#!/bin/bash
# Retain API examples for Hindsight CLI
# Run: bash examples/api/retain.sh

set -e

HINDSIGHT_URL="${HINDSIGHT_API_URL:-http://localhost:8888}"

# =============================================================================
# Doc Examples
# =============================================================================

# [docs:retain-basic]
hindsight memory retain my-bank "Alice works at Google as a software engineer"
# [/docs:retain-basic]


# [docs:retain-with-context]
hindsight memory retain my-bank "Alice got promoted" \
    --context "career update"
# [/docs:retain-with-context]


# [docs:retain-async]
hindsight memory retain my-bank "Meeting notes" --async
# [/docs:retain-async]


# [docs:retain-files]
# Upload a single file (PDF, DOCX, PPTX, XLSX, images, audio, and more)
hindsight memory retain-files my-bank report.pdf

# Upload a directory of files
hindsight memory retain-files my-bank ./documents/

# Upload and wait for processing to complete (polls until done)
hindsight memory retain-files my-bank report.pdf

# Queue files for background processing (returns immediately)
hindsight memory retain-files my-bank ./documents/ --async
# [/docs:retain-files]


# [docs:retain-files-curl]
# Via HTTP API (multipart/form-data)
curl -X POST "${HINDSIGHT_URL}/v1/default/banks/my-bank/files/retain" \
    -F "files=@report.pdf;type=application/octet-stream" \
    -F "request={\"async\": true, \"files_metadata\": [{\"context\": \"quarterly report\"}]}"
# [/docs:retain-files-curl]


# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
curl -s -X DELETE "${HINDSIGHT_URL}/v1/default/banks/my-bank" > /dev/null

echo "retain.sh: All examples passed"
