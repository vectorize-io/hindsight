---
title: "Gemini Spark Persistent Memory: Agent-Initiated Recall Over MCP"
authors: [benfrank241]
slug: "2026/06/15/gemini-spark-persistent-memory"
date: 2026-06-15T12:00
tags: [gemini-spark, memory, persistent-memory, hindsight, agents, mcp, tutorial]
description: "Add persistent long-term memory to Google's Gemini Spark assistant with Hindsight. No plugin host, no hooks. Spark calls Hindsight's recall and retain tools over MCP, with Hindsight Cloud or a self-hosted OAuth proxy."
image: /img/blog/gemini-spark-persistent-memory.png
hide_table_of_contents: true
---

![Gemini Spark Persistent Memory with Hindsight](/img/blog/gemini-spark-persistent-memory.png)

[Gemini Spark](https://gemini.google/overview/agent/spark/) is Google's always-on agentic assistant. It plans, calls tools, and works across your day. What it doesn't do on its own is carry anything from one conversation into the next. Ask it to remember a decision today and it's gone tomorrow.

:::note Spark is rolling out
As of this writing, Gemini Spark is **"coming soon."** Google describes it as *rolling out to trusted testers*, with availability for Google AI Ultra subscribers (18+, US) and select business users. The Hindsight integration below is ready now and targets that early-access audience; if you don't have Spark access yet, treat this as a guide to how memory will work once you do. The MCP config shapes follow Google's I/O 2026 developer guidance and may shift when the formal schema lands.
:::

This post is a walkthrough of the new Hindsight integration for Gemini Spark. It's a different shape than most of our integrations: there are no lifecycle hooks and no plugin code running next to the agent. Spark lives entirely on Google's cloud, so the only way in is **MCP**, and the integration is a config snippet that registers Hindsight as an MCP server Spark can call.

## TL;DR

<!-- truncate -->

- Gemini Spark has no persistent memory across sessions. Every conversation starts cold.
- The Hindsight integration is **config-only**. There's nothing to `pip install` and no daemon that runs beside Spark. You register Hindsight as an MCP server and Spark calls its `recall` and `retain` tools.
- **Memory is agent-initiated, not hook-driven.** Unlike the Cline or Claude Code integrations, Spark decides when to recall and when to retain, because third parties can't see Spark's prompt assembly or its transcripts.
- **Hindsight Cloud is the simplest path:** point Spark at `https://api.hindsight.vectorize.io/mcp` with a Bearer token, done.
- **Self-hosting works too,** but Spark only speaks OAuth 2.1 to MCP servers, so you put the [`cloudflare-oauth-proxy`](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/cloudflare-oauth-proxy) in front of your instance.
- The bank Spark writes to is a normal Hindsight bank. The same memories are readable from Claude Code, Cline, the API, or any other integration.

## Why Spark Is Different

Most of our coding-agent integrations work by running code inside the agent's host. Cline and Cursor expose lifecycle hooks, so Hindsight can recall context before each task automatically and retain the transcript when the task ends. That makes memory **deterministic**: it happens whether or not the model thinks to ask for it.

Spark gives you none of that. It runs on Google's infrastructure, reached through Antigravity 2.0, and there is no plugin host where Hindsight code can run alongside the agent loop. The table below is the honest accounting of what's possible:

| Capability | Spark support |
|---|---|
| Hook-based auto-recall (prepend context to the prompt) | Not available. Spark's prompt assembly is private. The agent calls `recall` when its planner judges it useful. |
| Hook-based auto-retain (save transcripts on turn end) | Not available. Third parties don't see Spark's transcripts. The agent calls `retain` when it learns something worth keeping. |
| MCP tools (`recall`, `retain`, and the rest) | Yes. Spark calls Hindsight's MCP tools through its built-in MCP client. |

So the integration leans entirely on the one surface Spark does expose: MCP. You register Hindsight's MCP server, and from then on `recall`, `retain`, and `reflect` are tools Spark can choose to use.

## How It Works

### With Hindsight Cloud (recommended)

The clean path. Spark talks straight to Hindsight Cloud over HTTPS using MCP's Streamable HTTP transport:

```
Gemini Spark (Google Cloud)
        |
        | HTTPS + MCP (Streamable HTTP)
        v
Hindsight Cloud (api.hindsight.vectorize.io)
```

No proxy, no tunnel, no infrastructure. The Cloud endpoint is already an OAuth-capable MCP server, and a Bearer token is enough to authenticate.

### With self-hosted Hindsight

If you run Hindsight yourself, there's one wrinkle: Spark only speaks **OAuth 2.1** to MCP servers. A raw `hindsight-embed` MCP server with a static token won't satisfy that handshake. The integration ships a Cloudflare Worker, [`cloudflare-oauth-proxy`](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/cloudflare-oauth-proxy), that acts as the OAuth authorization server Spark expects and bridges the authenticated request to your Hindsight API token:

```
Gemini Spark (Google Cloud)
        |
        | HTTPS + OAuth 2.1 (Spark's MCP client)
        v
Cloudflare Worker (cloudflare-oauth-proxy)
   - OAuth authorization server
   - Auth bridging to Hindsight API token
        |
        | HTTPS + Cloudflare Tunnel
        v
Self-hosted Hindsight + hindsight-embed (MCP server)
```

For most people, Cloud removes all of this. The proxy exists for teams that need memory to stay inside their own infrastructure.

## Setup

### Option 1: Hindsight Cloud

1. [Sign up free](https://ui.hindsight.vectorize.io/signup) and create a memory bank.
2. Copy your API key from the dashboard.
3. Register Hindsight in Spark's MCP config (below).

### Option 2: Self-hosted

1. Deploy a Hindsight instance and run the `hindsight-embed` MCP server pointed at it, exposed on a public HTTPS endpoint.
2. Deploy the [`cloudflare-oauth-proxy`](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/cloudflare-oauth-proxy), since Spark only speaks OAuth 2.1 to MCP servers.
3. Register the proxy URL in Spark's MCP config (below).

### Register Hindsight in Spark

Spark, via Antigravity 2.0, reads MCP servers from one of two places depending on how you run it.

**Antigravity desktop / IDE** reads `~/.gemini/antigravity/mcp_config.json`. Drop in:

```json
{
  "mcpServers": {
    "hindsight": {
      "serverUrl": "https://api.hindsight.vectorize.io/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_HINDSIGHT_API_KEY"
      }
    }
  }
}
```

**Hosted agent / Spark cloud** reads an `antigravity.yaml` manifest. Add a `tools.mcp_servers` entry:

```yaml
tools:
  mcp_servers:
    - name: hindsight
      endpoint: https://api.hindsight.vectorize.io/mcp
      auth: bearer
      description: >
        Long-term memory across sessions. Call recall whenever the user
        references past work, decisions, or preferences from earlier
        conversations. Call retain whenever the user shares a fact,
        preference, or decision worth remembering.
```

For self-hosting, swap the endpoint for your `cloudflare-oauth-proxy` URL. Both example files live in the [integration directory](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/gemini-spark): [`mcp_config.example.json`](https://github.com/vectorize-io/hindsight/blob/main/hindsight-integrations/gemini-spark/mcp_config.example.json) and [`manifest.example.yaml`](https://github.com/vectorize-io/hindsight/blob/main/hindsight-integrations/gemini-spark/manifest.example.yaml).

A note on the manifest: its shape follows the developer guidance Google published with Antigravity 2.0. Once the formal schema lands, the manifest field names may need a small adjustment. The MCP config for the desktop client is the more stable of the two today.

## The Tool Description Is the Prompt

Because there are no hooks, the agent's behavior is steered entirely by the **MCP tool descriptions** and the manifest's `description` field. This is the one knob that matters most, so it's worth dwelling on.

When Spark plans a turn, it scans the tools it has and their descriptions to decide what to call. A vague description ("memory tool") gets ignored at exactly the moments you want it used. The example manifest deliberately spells out the triggers:

> Call recall whenever the user references past work, decisions, or preferences from earlier conversations. Call retain whenever the user shares a fact, preference, or decision worth remembering.

If you want Spark to lean on memory more aggressively, sharpen that text. Tell it to recall before answering questions about ongoing projects, or to retain after the user states a constraint. You're not configuring Hindsight here so much as instructing Spark's planner about when memory is relevant.

## Verifying It Works

There's no debug log on the agent side to watch, so the test is behavioral. Prompt Spark with something that should obviously route to a memory tool:

- **Recall:** "What were my open API decisions from last week?" should trigger a `recall` call.
- **Retain:** "Remember that I prefer TypeScript strict mode for new projects." should trigger a `retain` call.

Then start a fresh conversation and ask Spark what it knows about your TypeScript preference. If memory is wired correctly, the second session recalls what the first one stored. You can also confirm from the other side: open your bank in the Hindsight dashboard and watch the retained memories appear.

## Tradeoffs

**Memory is best-effort, not guaranteed.** This is the headline tradeoff and the reason this integration reads differently from the hook-based ones. With Cline, recall runs on every task whether the model wants it or not. With Spark, recall happens only when the planner decides to call the tool. Good tool descriptions get you most of the way, but a model that doesn't think to recall won't.

**You can't see Spark's transcript.** Retain only captures what the agent chooses to pass into the `retain` tool, not the full conversation. If Spark summarizes a decision in one sentence and stores that, that sentence is what you get. There's no transcript-accumulation step like the one the Cline integration runs, because the transcript isn't exposed.

**Self-hosting adds an OAuth hop.** The Cloudflare proxy is a real piece of infrastructure to deploy and keep alive. If you don't have a hard requirement to self-host, Cloud skips it entirely.

## Recap

| | Spark default | With Hindsight (MCP) |
| --- | --- | --- |
| Memory across sessions | None | Persistent, per bank |
| Setup | n/a | Config-only: register one MCP server |
| Recall mechanism | n/a | Agent-initiated `recall` tool call |
| Retain mechanism | n/a | Agent-initiated `retain` tool call |
| Determinism | n/a | Best-effort (planner decides), tuned via tool descriptions |
| Cross-tool sharing | n/a | Same bank readable from Claude Code, Cline, the API |

## Next Steps

- **Hindsight Cloud:** [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io/signup)
- **Integration docs:** [Gemini Spark + Hindsight](/sdks/integrations/gemini-spark)
- **Source:** [vectorize-io/hindsight/hindsight-integrations/gemini-spark](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/gemini-spark)
- **OAuth proxy:** [cloudflare-oauth-proxy](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/cloudflare-oauth-proxy)
- **Why MCP for memory:** [Native OAuth for MCP Clients](https://hindsight.vectorize.io/blog/2026/03/27/mcp-oauth-native)
