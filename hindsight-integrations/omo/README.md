# Hindsight × OMO (oh-my-openagent)

Long-term memory for [OMO](https://github.com/code-yeongyu/oh-my-openagent) agents via [Hindsight](https://hindsight.vectorize.io). Automatically recalls relevant context before each prompt and retains session learnings for future use.

## Setup

### 1. Get an API key

Sign up at [hindsight.vectorize.io](https://hindsight.vectorize.io) and create an API key (`hsk_...`).

### 2. Install the integration

From the `hindsight-integrations/omo/` directory:

```bash
# Hooks (global)
mkdir -p ~/.omo/hooks
cp hooks/hooks.json ~/.omo/hooks/hindsight-hooks.json

# Scripts + settings (global)
mkdir -p ~/.omo/plugins/hindsight/scripts
cp -r scripts/ ~/.omo/plugins/hindsight/scripts/
cp settings.json ~/.omo/plugins/hindsight/settings.json

# Rules (per-project — run from your project root)
mkdir -p /path/to/your/project/.omo/rules
cp rules/hindsight-memory.md /path/to/your/project/.omo/rules/hindsight-memory.md
```

### 3. Set your API key

```bash
export HINDSIGHT_API_TOKEN=hsk_your_key_here
```

Or persistently in `~/.hindsight/omo.json`:

```json
{
  "hindsightApiToken": "hsk_your_key_here"
}
```

### 4. Allow env vars in OMO config

In `~/.config/opencode/oh-my-openagent.jsonc`:

```jsonc
{
  "mcp_env_allowlist": [
    "HINDSIGHT_API_URL",
    "HINDSIGHT_API_TOKEN",
    "HINDSIGHT_BANK_ID"
  ]
}
```

That's it. Start OMO and memory works automatically.

## Self-Hosted (optional)

To use a self-hosted Hindsight instance instead of cloud, override the API URL:

```bash
export HINDSIGHT_API_URL=http://localhost:8888
```

Or in `~/.hindsight/omo.json`:

```json
{
  "hindsightApiUrl": "http://localhost:8888",
  "hindsightApiToken": null
}
```

No API token is required for local instances.

## How It Works

| Hook Event | When | Action |
|---|---|---|
| `SessionStart` | Session begins | Health check; warn if API key missing |
| `UserPromptSubmit` | Before each prompt | Query Hindsight for relevant memories; inject as context |
| `Stop` | Agent finishes | Extract transcript; send to Hindsight for fact extraction |
| `SubagentStop` | Sub-agent finishes | Same as Stop — captures sub-agent learnings |
| `SessionEnd` | Session terminates | Force final retain for short sessions |

All hooks degrade gracefully: if Hindsight is unreachable, OMO continues working normally without memory.

## Configuration

Settings are loaded in order (later wins):

1. `settings.json` (plugin defaults — cloud URL pre-set)
2. `~/.hindsight/omo.json` (user overrides)
3. `HINDSIGHT_*` environment variables

### Key Settings

| Setting | Env Var | Default | Description |
|---|---|---|---|
| `hindsightApiUrl` | `HINDSIGHT_API_URL` | `https://api.hindsight.vectorize.io` | API endpoint |
| `hindsightApiToken` | `HINDSIGHT_API_TOKEN` | — | API key (`hsk_...`), required for cloud |
| `bankId` | `HINDSIGHT_BANK_ID` | `omo` | Memory bank name |
| `autoRecall` | `HINDSIGHT_AUTO_RECALL` | `true` | Auto-recall before prompts |
| `autoRetain` | `HINDSIGHT_AUTO_RETAIN` | `true` | Auto-retain after responses |
| `retainEveryNTurns` | — | `10` | Retain frequency (turns) |
| `recallBudget` | `HINDSIGHT_RECALL_BUDGET` | `mid` | Recall depth (`low`/`mid`/`high`) |
| `dynamicBankId` | `HINDSIGHT_DYNAMIC_BANK_ID` | `false` | Per-project bank isolation |
| `debug` | `HINDSIGHT_DEBUG` | `false` | Enable debug logging to stderr |

### Dynamic Bank IDs

Enable per-project memory isolation:

```json
{
  "dynamicBankId": true,
  "dynamicBankGranularity": ["agent", "project"]
}
```

This creates banks like `omo::myproject`, `omo::other-repo`, etc.

### Multi-Bank Recall

Query additional banks alongside the primary one:

```json
{
  "recallAdditionalBanks": ["shared-team-knowledge"]
}
```

## Architecture

```
OMO (orchestrator)
 ├── SessionStart hook → health check
 ├── UserPromptSubmit hook → recall memories → inject as additionalContext
 ├── Stop hook → retain session transcript (async)
 ├── SubagentStop hook → retain sub-agent findings (async)
 └── SessionEnd hook → force final retain
```

## Testing

```bash
# Run unit tests
cd hindsight-integrations/omo
pip install pytest
python -m pytest tests/ -v

# Run interactive demo against a local Hindsight server
HINDSIGHT_API_URL=http://localhost:8888 python demo.py
```
