# Hindsight × Gemini Spark

Long-term memory for [Gemini Spark](https://blog.google/products/gemini/gemini-spark/),
Google's always-on agentic assistant, via MCP.

> **Recommended:** Use [Hindsight Cloud](https://vectorize.io/hindsight/) for the
> fastest setup — no infrastructure to manage. The setup below works with both
> Cloud and self-hosted Hindsight.

## How it works

Spark runs on Google's cloud infrastructure. Unlike OpenClaw or Claude Code,
there is **no plugin host** where Hindsight code runs alongside Spark's agent
loop. The only third-party extension surface is MCP:

| Capability | Spark support |
|---|---|
| Hook-based auto-recall (prepend context to prompt) | Not available — Spark's prompt assembly is private. The agent calls `recall` when its planner judges it useful. |
| Hook-based auto-retain (save transcripts on turn end) | Not available — third parties don't see Spark's transcripts. The agent calls `retain` when it learns something worth keeping. |
| MCP tools (`recall`, `retain`, etc.) | Yes — Spark calls Hindsight's MCP tools via its built-in MCP client. |

## Architecture

### With Hindsight Cloud (recommended)

```
Gemini Spark (Google Cloud)
        |
        | HTTPS + MCP (Streamable HTTP)
        v
Hindsight Cloud (api.hindsight.vectorize.io)
```

### With self-hosted Hindsight

```
Gemini Spark (Google Cloud)
        |
        | HTTPS + OAuth 2.1 (Spark's MCP client)
        v
Cloudflare Worker — hindsight-integrations/cloudflare-oauth-proxy
   - OAuth authorization server
   - Auth bridging to Hindsight API token
        |
        | HTTPS + Cloudflare Tunnel
        v
Self-hosted Hindsight + hindsight-embed (MCP server)
```

## Setup

### Option 1: Hindsight Cloud (recommended)

1. Sign up at [vectorize.io/hindsight](https://vectorize.io/hindsight/) and
   create a memory bank
2. Copy your API key from the dashboard
3. Register Hindsight in Spark's MCP config (see below)

### Option 2: Self-hosted

1. Deploy a Hindsight instance — see the
   [self-hosting quickstart](https://vectorize.io/hindsight/docs/quickstart/self-hosting)
2. Run the `hindsight-embed` MCP server pointed at your instance, exposed on
   a public HTTPS endpoint
3. Deploy the [`cloudflare-oauth-proxy`](../cloudflare-oauth-proxy/) (Spark
   only speaks OAuth 2.1 to MCP servers)

### Register Hindsight in Spark

Spark (via Antigravity 2.0) reads MCP servers from one of two places:

- **Hosted agent / Spark cloud:** an `antigravity.yaml` manifest — see
  [`manifest.example.yaml`](./manifest.example.yaml)
- **Antigravity desktop / IDE:** `~/.gemini/antigravity/mcp_config.json` —
  see [`mcp_config.example.json`](./mcp_config.example.json)

Replace the placeholder URL with your Hindsight Cloud endpoint or OAuth proxy
URL.

### Verify it works

Prompt Spark with something that should trigger memory tools:

- "What were my open API decisions from last week?" → `recall`
- "Remember that I prefer TypeScript strict mode for new projects." →
  `retain`

## Example configs

- [`manifest.example.yaml`](./manifest.example.yaml) — Antigravity 2.0 agent
  manifest snippet
- [`mcp_config.example.json`](./mcp_config.example.json) — Desktop/IDE MCP
  config for local development
