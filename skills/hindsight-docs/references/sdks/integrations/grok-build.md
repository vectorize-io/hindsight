
# Grok Build

Biomimetic long-term memory for [Grok Build](https://x.ai/cli) using [Hindsight](https://vectorize.io/hindsight). Automatically captures conversations and recalls relevant context across sessions — no changes to your workflow required.

> **💡 Powered by the Claude Code plugin**
>
Grok Build natively reads Claude Code plugin format — hooks, MCP servers, skills, and marketplace metadata all work without modification. This integration uses the same [`hindsight-memory` plugin](claude-code.md) that powers Claude Code. All features, configuration options, and knowledge tools are fully available in Grok Build.
## Quick Start

Grok Build reads Claude Code plugins natively, so installation uses the standard Claude Code commands. Grok Build will discover and activate the plugin automatically.

```bash
# 1. Add the Hindsight marketplace and install the plugin
claude plugin marketplace add vectorize-io/hindsight
claude plugin install hindsight-memory

# 2. Configure your LLM provider for memory extraction
# Option A: OpenAI (auto-detected)
export OPENAI_API_KEY="sk-your-key"

# Option B: Anthropic (auto-detected)
export ANTHROPIC_API_KEY="your-key"

# Option C: Connect to an external Hindsight server instead of running locally
mkdir -p ~/.hindsight
echo '{"hindsightApiUrl": "https://your-hindsight-server.com"}' > ~/.hindsight/claude-code.json

# 3. Start Grok Build — the plugin activates automatically
grok
```

That's it! The plugin will automatically start capturing and recalling memories.

## Features

- **Auto-recall** — on every user prompt, queries Hindsight for relevant memories and injects them as context (invisible to the chat transcript, visible to Grok)
- **Auto-retain** — after every response (or every N turns), extracts and retains conversation content for long-term storage
- **Knowledge tools** — Grok can read, write, and search its own memory via MCP tools (`agent_knowledge_recall`, `agent_knowledge_ingest`, etc.)
- **Dynamic bank IDs** — per-agent, per-project, or per-session memory isolation
- **Daemon management** — can auto-start/stop `hindsight-embed` locally or connect to an external Hindsight server

## Architecture

The plugin hooks into Grok Build's lifecycle events:

| Component | Trigger | Purpose |
|-----------|---------|---------|
| `session_start.py` | `SessionStart` hook | Health check — verify Hindsight is reachable |
| `recall.py` | `UserPromptSubmit` hook | **Auto-recall** — query memories, inject as `additionalContext` |
| `retain.py` | `Stop` hook | **Auto-retain** — extract transcript, POST to Hindsight (async) |
| `session_end.py` | `SessionEnd` hook | Cleanup — stop auto-managed daemon if started |
| `mcp_server.py` | MCP server | Exposes `agent_knowledge_*` tools — list/get/create/update/delete pages, recall, ingest |

## Configuration

The plugin reads configuration from `~/.hindsight/claude-code.json` — the same file used by Claude Code, regardless of which host (Grok Build or Claude Code) is running the plugin.

**Loading order** (later entries win):
1. Built-in defaults
2. Plugin `settings.json` (ships with the plugin)
3. User config (`~/.hindsight/claude-code.json`)
4. Environment variables (`HINDSIGHT_*`)

For the full configuration reference — connection settings, LLM provider, memory bank, auto-recall, auto-retain, knowledge tools, and debug options — see the [Claude Code configuration docs](claude-code.md#configuration).

### Separating Grok Build and Claude Code memory

Both tools share `~/.hindsight/claude-code.json`, so by default they share memory. If you want separate memory banks for each tool, override the agent name via environment variables in each tool's startup environment:

```bash
# In your Grok Build shell session
export HINDSIGHT_AGENT_NAME=grok-build
export HINDSIGHT_BANK_ID=grok_build
grok
```

With `dynamicBankId` enabled in your config, this produces bank IDs like `grok-build::myproject` instead of `claude-code::myproject`, fully isolating memory between the two tools.

## Per-Project Memory

To give each project its own isolated memory bank, set this in `~/.hindsight/claude-code.json`:

```json
{
  "dynamicBankId": true,
  "dynamicBankGranularity": ["agent", "project"]
}
```

With this config, running Grok Build in `~/projects/api` and `~/projects/frontend` stores and recalls memories separately. Git worktrees of the same repo share a bank by default.

## Troubleshooting

**Plugin not listed**: Run `grok plugin list` to see installed plugins. If `hindsight-memory` is missing, re-run the install command.

**Hooks not firing**: Run `grok inspect` and check the Hooks section for `hindsight-memory`. Enable `"debug": true` in your config to see `[Hindsight]` messages in stderr.

**No memories recalled**: Memories need at least one retain cycle before they're available. Complete a full session first (say something, exit, start a new session).

**High latency on recall**: Use `"recallBudget": "low"` or reduce `recallMaxTokens` for faster responses.

**Debug mode**: Add `"debug": true` to your config file:

```
[Hindsight] Recalling from bank 'grok-build::myproject', query length: 42
[Hindsight] Injecting 3 memories
[Hindsight] Retaining to bank 'grok-build::myproject', doc 'sess-abc123', 2 messages, 847 chars
```

**State files**: Plugin state is stored at `~/.grok/plugins/data/hindsight-memory/state/`. Check `last_recall.json` to see what was most recently recalled.
