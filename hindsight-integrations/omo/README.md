# Hindsight × OMO (oh-my-openagent)

Long-term memory for [OMO](https://github.com/code-yeongyu/oh-my-openagent) agents via [Hindsight](https://hindsight.vectorize.io). Automatically recalls relevant context before each prompt and retains session learnings for future use.

## Quick Start (Hindsight Cloud)

1. **Get an API key** at [ui.hindsight.vectorize.io/signup](https://ui.hindsight.vectorize.io/signup)

2. **Set your API key:**
   ```bash
   export HINDSIGHT_API_TOKEN=hsk_your_key_here
   ```

3. **Copy the integration files** into your OMO setup:
   ```bash
   # Copy hooks
   cp hooks/hooks.json ~/.omo/hooks/hindsight-hooks.json

   # Copy rules
   cp rules/hindsight-memory.md .omo/rules/hindsight-memory.md

   # Copy scripts
   cp -r scripts/ ~/.omo/plugins/hindsight/scripts/
   cp settings.json ~/.omo/plugins/hindsight/settings.json
   ```

4. **Add env vars to OMO's allowlist** (in `~/.config/opencode/oh-my-openagent.jsonc`):
   ```jsonc
   {
     "mcp_env_allowlist": [
       "HINDSIGHT_API_URL",
       "HINDSIGHT_API_TOKEN",
       "HINDSIGHT_BANK_ID"
     ]
   }
   ```

## Self-Hosted

To use a self-hosted Hindsight instance instead of cloud:

```bash
export HINDSIGHT_API_URL=http://localhost:8888
# API token is optional for local instances
```

Or in `~/.hindsight/omo.json`:
```json
{
  "hindsightApiUrl": "http://localhost:8888",
  "hindsightApiToken": null
}
```

## How It Works

| Hook Event | When | Action |
|---|---|---|
| `SessionStart` | Session begins | Health check; warn if API key missing |
| `UserPromptSubmit` | Before each prompt | Query Hindsight for relevant memories; inject as context |
| `Stop` | Agent finishes | Extract transcript; send to Hindsight for fact extraction |
| `SubagentStop` | Sub-agent finishes | Same as Stop — captures sub-agent learnings |
| `SessionEnd` | Session terminates | Force final retain for short sessions |

## Configuration

Settings are loaded in order (later wins):

1. `settings.json` (plugin defaults — cloud URL pre-set)
2. `~/.hindsight/omo.json` (user overrides)
3. Environment variables

### Key Settings

| Setting | Env Var | Default | Description |
|---|---|---|---|
| `hindsightApiUrl` | `HINDSIGHT_API_URL` | `https://api.hindsight.vectorize.io` | API endpoint |
| `hindsightApiToken` | `HINDSIGHT_API_TOKEN` | — | API key (`hsk_...`) |
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

The integration is **cloud-first** — no local daemon management needed. For self-hosted setups, just point `HINDSIGHT_API_URL` to your instance.

All hooks degrade gracefully: if Hindsight is unreachable, OMO continues working normally without memory.
