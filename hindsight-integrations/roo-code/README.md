# hindsight-roo-code

Persistent long-term memory for [Roo Code](https://github.com/RooVetGit/Roo-Code) via [Hindsight](https://github.com/vectorize-io/hindsight).

Run the installer once and every Roo Code session automatically recalls past context before tasks and retains learnings after.

## What It Does

- **Before each task** — Roo Code recalls relevant memories from Hindsight and includes them as context
- **During a task** — agents can store important decisions and discoveries immediately via `hindsight_retain`
- **After each task** — agents summarize and retain what was accomplished for future sessions

## Prerequisites

A running Hindsight instance. Either:

```bash
# Self-hosted (runs locally on port 8888)
pip install hindsight-all
export HINDSIGHT_API_LLM_API_KEY=your-openai-key
hindsight-api
```

Or [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) — no self-hosting required.

## Installation

From your project directory:

```bash
python /path/to/hindsight-integrations/roo-code/install.py
```

Or with a custom API URL:

```bash
python install.py --api-url https://my-hindsight.example.com
```

Install globally (applies to all projects):

```bash
python install.py --global
```

This writes two files:
- `.roo/mcp.json` — registers the Hindsight MCP server with Roo Code
- `.roo/rules/hindsight-memory.md` — rules injected into every system prompt

## How It Works

```
New task starts
  └─ Roo Code rules instruct agent to call hindsight_recall
       └─ Relevant memories injected into context

Agent working…
  └─ hindsight_retain stores decisions/discoveries immediately

Task ends
  └─ Roo Code rules instruct agent to call hindsight_retain with summary
       └─ Summary stored for future sessions
```

Hindsight exposes `hindsight_recall` and `hindsight_retain` as MCP tools via its built-in `/mcp` endpoint. The rules file in `.roo/rules/` tells Roo Code when to call them.

## Configuration

The installer writes to `.roo/mcp.json`:

```json
{
  "mcpServers": {
    "hindsight": {
      "url": "http://localhost:8888/mcp",
      "timeout": 10000,
      "alwaysAllow": ["hindsight_recall", "hindsight_retain"]
    }
  }
}
```

To change the API URL after installation, edit `.roo/mcp.json` directly or re-run the installer:

```bash
python install.py --api-url https://new-url.example.com
```

## Verifying Setup

1. Start Hindsight (`hindsight-api` or Hindsight Cloud)
2. Open Roo Code in your project
3. Check **Settings → MCP Servers** — `hindsight` should show as connected
4. Start a task — you should see `hindsight_recall` called automatically

## Development

```bash
pip install pytest
python -m pytest tests/ -v
```
