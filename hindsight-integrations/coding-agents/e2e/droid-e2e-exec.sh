#!/bin/sh
# Render Droid's settings from the environment, then run one prompt.
#
# Two things cannot be passed to `droid exec` as flags or environment:
#
#   * the model route — Droid reads `baseUrl` from settings.json and expands `${VAR}` only in
#     `apiKey`, so the stub's ephemeral port has to be written into the file at run time;
#   * workspace trust — a fresh container has never trusted /workspace, and the non-interactive
#     path cannot show the trust prompt.
#
# Writing settings.json here (after run-harness.sh has already run the installer) is safe: the
# installer owns hooks.json, mcp.json and skills/, and only ever READS settings.json — for the
# fallback hook declarations it copies when creating hooks.json.
set -eu

: "${HINDSIGHT_STUB_BASE_URL:?missing stub model base URL}"

mkdir -p /root/.factory
cat > /root/.factory/settings.json <<EOF
{
  "customModels": [
    {
      "model": "hindsight-e2e-stub",
      "displayName": "Hindsight E2E stub",
      "baseUrl": "${HINDSIGHT_STUB_BASE_URL}",
      "apiKey": "hindsight-e2e",
      "provider": "generic-chat-completion-api",
      "maxOutputTokens": 8192
    }
  ],
  "trustedFolders": {
    "/workspace": {"trustedAt": "2026-01-01T00:00:00.000Z"}
  }
}
EOF

exec droid exec --model hindsight-e2e-stub --cwd /workspace "$@"
