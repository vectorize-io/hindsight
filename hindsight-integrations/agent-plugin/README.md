# Hindsight — Agent Plugin

A portable [Agent Plugin](https://agent-plugins.org) (spec `1.0.0`) that gives any
compatible agent client long-term memory via [Hindsight](https://hindsight.vectorize.io).

Agent Plugins is the vendor-neutral standard (AWS, Cursor, GitHub/Microsoft, OpenAI,
Vercel) for packaging **Agent Skills + MCP servers** into one distributable plugin. At
launch it is supported by **ChatGPT/Codex, Cursor, GitHub Copilot, Kiro, and VS Code**.

This is the *portable* front door to Hindsight: one artifact, every supported client. It
carries the same `retain` / `recall` / `reflect` memory as our per-IDE integrations, but
as a single standards-based bundle instead of N hand-rolled configs.

## What's in the bundle

```
agent-plugin/
├── plugin.json                       # manifest ($schema + name + metadata)
├── mcp.json                          # Hindsight MCP server (Streamable HTTP)
└── skills/
    └── hindsight-memory/
        └── SKILL.md                  # teaches the agent when to recall/retain/reflect
```

- **`mcp.json`** points the client at Hindsight's built-in MCP server (retain, recall,
  reflect, knowledge pages, and more — see [MCP Server docs](https://hindsight.vectorize.io/developer/mcp-server)).
  The plugin is transport-only; all memory logic stays server-side.
- **`skills/hindsight-memory/SKILL.md`** is loaded into the agent's context so it knows
  *when* to reach for memory, not just that the tools exist.

## Configuration

The plugin reads two environment variables (values are interpolated into `mcp.json`):

| Variable | Required | Purpose |
|----------|----------|---------|
| `HINDSIGHT_API_KEY` | yes (Cloud) | Your `hsk_...` key from [ui.hindsight.vectorize.io/connect](https://ui.hindsight.vectorize.io/connect). Sent as `Authorization: Bearer`. |
| `HINDSIGHT_BANK_ID` | no | Memory bank to scope to (sent as `X-Bank-Id`). Defaults to `default`. Use one bank per user/project/team. |

**Self-hosting:** replace the host in `mcp.json` (`https://api.hindsight.vectorize.io`)
with your deployment's URL. A local server with the MCP endpoint open needs no API key.

> Env-var interpolation syntax varies by client. Most use `${VAR}`; some (VS Code,
> Cursor) prefer `${env:VAR}`. If your client doesn't substitute, paste the literal key
> and bank id into `mcp.json` instead.

## Install

Installation and distribution are intentionally left to each client by the spec. Common
paths:

- **VS Code / GitHub Copilot / Cursor / Kiro** — add this plugin directory through the
  client's plugin/MCP UI, or drop it where the client discovers plugins, then set the
  two environment variables above.
- **Codex / ChatGPT** — register the plugin per the client's Agent Plugins support.

Once installed, ask the agent something that depends on past context (or tell it a
durable preference) and it will call `recall` / `retain` automatically, guided by the
skill.

## Want automatic capture (hooks)?

Agent Plugins `1.0.0` standardizes **Skills + MCP**, not session lifecycle hooks. This
plugin therefore delivers **explicit, tool-driven** memory that works identically
everywhere. For the fully automatic experience — recall injected before every prompt and
transcripts retained on session end — use the native, hook-based integration for your
tool (e.g. [`hindsight-integrations/claude-code`](../claude-code)). The two share the
same banks, so memory captured by one is recalled by the other.

## Validate the manifests

```bash
python3 hindsight-integrations/agent-plugin/validate.py
```

Checks that `plugin.json` and `mcp.json` parse and satisfy the Agent Plugins `1.0.0`
required-field contract.
