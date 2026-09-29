#!/bin/sh
# Point ZCode at the stub model, then run one prompt.
#
# ZCode takes its provider from ~/.zcode/cli/config.json and from nothing else — no base-URL
# environment variable, no flag — so the stub's ephemeral port has to be written into the file at
# run time, the same problem Droid and dsh have.
#
# What is different here, and why this MERGES instead of rendering the file like droid-e2e-exec.sh:
# that config is the SAME file `hindsight-coding-agents install zcode` just wrote the hooks block
# and the MCP server into. Overwriting it would delete the wiring this E2E exists to test, and the
# run would pass for the wrong reason (no hooks, so nothing to break).
#
# `apiKeyRequired` plus any non-empty key satisfies the upstream login gate, so no Z.AI account is
# needed; the provider key must be `zai` for that gate when it is the only provider.
set -eu

: "${HINDSIGHT_STUB_BASE_URL:?missing stub model base URL}"

CONFIG=/root/.zcode/cli/config.json
mkdir -p "$(dirname "$CONFIG")"
node -e '
const fs = require("node:fs");
const [path, baseURL] = process.argv.slice(1);
const config = fs.existsSync(path) ? JSON.parse(fs.readFileSync(path, "utf8")) : {};
config.provider = {
  zai: {
    kind: "anthropic",
    name: "Hindsight E2E stub",
    options: { baseURL, apiKey: "hindsight-e2e", apiKeyRequired: true },
    models: { "hindsight-e2e-stub": { name: "Hindsight E2E stub" } },
  },
};
config.model = { main: "zai/hindsight-e2e-stub", lite: "zai/hindsight-e2e-stub" };
fs.writeFileSync(path, JSON.stringify(config, null, 2));
' "$CONFIG" "$HINDSIGHT_STUB_BASE_URL"

exec zcode --prompt "$1" --cwd /workspace
