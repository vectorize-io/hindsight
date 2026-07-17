---
title: "Guide: Add Gemini Spark Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, gemini-spark, agents, memory]
description: "Add Gemini Spark memory with Hindsight by registering Hindsight's MCP server, so Spark can recall relevant context and retain what it learns across sessions."
image: /img/guides/guide-gemini-spark-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add Gemini Spark Memory with Hindsight](/img/guides/guide-gemini-spark-memory-with-hindsight.svg)

If you want **Gemini Spark memory with Hindsight**, the setup is to register Hindsight's MCP server in Spark's config. Spark runs on Google's cloud infrastructure, so there is no plugin host where Hindsight code runs alongside Spark's agent loop. The only third-party extension surface is MCP, and Spark has a built-in MCP client that can call Hindsight's `recall` and `retain` tools.

This is a good fit for Spark because its prompt assembly and transcripts are private — third parties don't see them — so hook-based auto-recall and auto-retain aren't available. Instead, Spark's planner calls `recall` when it judges that past context is useful, and calls `retain` when it learns something worth keeping. That gives Spark long-term memory across sessions through the tools it already knows how to invoke.

This guide walks through signing up (or self-hosting), registering Hindsight in Spark's MCP config, and a quick verification flow so you can confirm the memory tools are actually being called. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. Sign up for [Hindsight Cloud](https://hindsight.vectorize.io) and create a memory bank (or self-host).
> 2. Copy your API key from the dashboard.
> 3. Register Hindsight as an MCP server in Spark's config with your endpoint and a Bearer token.
> 4. Replace the placeholder URL with your Hindsight Cloud endpoint (or OAuth proxy URL if self-hosted).
> 5. Verify by prompting Spark so its planner calls `recall` or `retain`.

## Prerequisites

Before you start, make sure you have:

- Access to Gemini Spark (via Antigravity 2.0)
- A reachable Hindsight backend, [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server
- Your Hindsight API key

## Step 1: Get a Hindsight backend

For Hindsight Cloud (recommended):

1. Sign up at [vectorize.io/hindsight](https://vectorize.io/hindsight/) and create a memory bank.
2. Copy your API key from the dashboard.

For self-hosted Hindsight:

1. Deploy a Hindsight instance and run the `hindsight-embed` MCP server pointed at it, exposed on a public HTTPS endpoint.
2. Deploy the [`cloudflare-oauth-proxy`](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/cloudflare-oauth-proxy), because Spark only speaks OAuth 2.1 to MCP servers.

## Step 2: Register Hindsight in Spark

Spark (via Antigravity 2.0) reads MCP servers from one of two places, depending on where Spark runs.

**Antigravity desktop / IDE** reads `~/.gemini/antigravity/mcp_config.json`. Add Hindsight as an MCP server there:

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

**Hosted agent / Spark cloud** reads an `antigravity.yaml` manifest. Drop the `tools.mcp_servers` block into your existing manifest:

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

Replace the placeholder URL with your Hindsight Cloud endpoint, or your `cloudflare-oauth-proxy` URL if you self-host.

## How Spark uses memory

Unlike editor integrations that hook the prompt directly, Spark's prompt assembly and transcripts are private, so recall and retain happen through Spark's planner rather than through hooks:

- **Recall:** Spark calls Hindsight's `recall` tool when its planner judges that past context — decisions, preferences, or earlier work — is useful for the current turn.
- **Retain:** Spark calls Hindsight's `retain` tool when it learns something worth keeping, such as a fact, preference, or decision.

Because both are model-driven tool calls rather than automatic hooks, the description text in the MCP config matters — it tells Spark's planner when to reach for each tool.

For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Verify that memory is working

Prompt Spark with something that should trigger the memory tools:

1. Ask Spark to remember something, for example: "Remember that I prefer TypeScript strict mode for new projects." This should trigger `retain`.
2. In a later turn, ask about it, for example: "What were my open API decisions from last week?" This should trigger `recall`.

If Spark calls the tools and surfaces the earlier information, the setup is working.

## Common mistakes

### Leaving the placeholder URL in place

The example configs ship with a placeholder endpoint. Replace it with your Hindsight Cloud endpoint, or your OAuth proxy URL if you self-host.

### Expecting automatic recall or retain

Spark has no plugin host and its transcripts are private, so there are no auto-recall or auto-retain hooks. Both are model-driven tool calls that Spark's planner decides to make.

### Skipping the OAuth proxy when self-hosting

Spark only speaks OAuth 2.1 to MCP servers. If you self-host, deploy the `cloudflare-oauth-proxy` in front of your `hindsight-embed` MCP server rather than pointing Spark directly at it.

### Registering in the wrong config location

Desktop/IDE reads `~/.gemini/antigravity/mcp_config.json`, while a hosted agent reads the `antigravity.yaml` manifest. Add Hindsight to the one that matches where Spark runs.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — run the `hindsight-embed` MCP server behind the `cloudflare-oauth-proxy` and point Spark at your proxy URL.

### Why doesn't memory recall automatically?

Spark's prompt assembly and transcripts are private, so third parties can't hook them. Recall and retain are tool calls that Spark's planner makes when it judges them useful.

### How do I authenticate to Hindsight Cloud?

With a Bearer token — put your Hindsight API key in the `Authorization` header of the MCP config.

### Where do I register the MCP server?

In `~/.gemini/antigravity/mcp_config.json` for the desktop/IDE, or in the `antigravity.yaml` manifest for a hosted agent.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [Gemini Spark integration docs](https://hindsight.vectorize.io/docs/integrations/gemini-spark)
