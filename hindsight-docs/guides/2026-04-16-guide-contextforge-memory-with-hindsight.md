---
title: "Guide: Set Up ContextForge Memory with Hindsight"
authors: [benfrank241]
date: 2026-04-16
tags: [how-to, contextforge, mcp, memory]
description: "Set up ContextForge memory with Hindsight so every client behind the gateway can use retain, recall, and reflect through one unified MCP endpoint."
image: /img/blog/guide-contextforge-memory-with-hindsight.png
hide_table_of_contents: true
---

![Guide: Set Up ContextForge Memory with Hindsight](/img/blog/guide-contextforge-memory-with-hindsight.png)

If you want **ContextForge memory with Hindsight**, the core idea is simple: register Hindsight as an MCP backend inside ContextForge, then let your AI tools connect to ContextForge instead of talking to Hindsight directly. That gives you one gateway, one authentication layer, and one place to expose long-term memory to every compatible client.

This is a strong pattern for teams because ContextForge already solves gateway problems like auth, RBAC, and central endpoint management. Hindsight adds the memory layer on top: retain for storing durable context, recall for searching it, and reflect for synthesizing what the system has learned over time.

This guide walks through the registration flow, explains when single-bank vs multi-bank mode matters, and shows how to verify that Hindsight tools are actually surfacing through ContextForge. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. Run Hindsight and ContextForge so they can reach each other.
> 2. Register Hindsight as an MCP server in ContextForge.
> 3. Choose whether ContextForge should expose Hindsight in single-bank or multi-bank mode.
> 4. Connect your client to ContextForge's endpoint, not directly to Hindsight.
> 5. Verify that retain, recall, and reflect are visible and callable.

## Prerequisites

Before you start, make sure you have:

- A running Hindsight deployment, either local, Kubernetes, or Cloud
- A running ContextForge instance
- Network connectivity from ContextForge to Hindsight
- Admin access to the ContextForge registration flow

If you are still deciding how to run Hindsight itself, start with [Hindsight Cloud](https://hindsight.vectorize.io) or the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) before adding the gateway layer.

## Why pair ContextForge with Hindsight

ContextForge is useful when you want a single MCP entry point for several systems. Hindsight is useful when your agents need durable memory instead of a stateless prompt loop. Put them together and you get:

- one authenticated endpoint for many MCP backends
- memory exposed alongside your other servers
- centralized access control
- the ability to scope memory by team or bank design

This is cleaner than wiring Hindsight directly into every downstream client one by one.

## Option 1: Register Hindsight in the ContextForge UI

Inside ContextForge:

1. Sign in as an admin.
2. Open **Servers**.
3. Click **Add Server**.
4. Fill in:
   - **Name**: `hindsight`
   - **URL**: your Hindsight MCP endpoint, usually `http://<host>:8888/mcp`
   - **Transport**: Streamable HTTP
5. Save the server.

That is the fastest path when you are testing or doing one-off setup.

## Option 2: Register Hindsight through the admin API

If you prefer automation, register the server through the ContextForge API.

```bash
TOKEN=$(curl -s -X POST https://your-contextforge.com/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@example.com", "password": "your-password"}' \
  | jq -r '.access_token')

curl -X POST https://your-contextforge.com/admin/servers \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "hindsight",
    "url": "http://hindsight-api:8888/mcp",
    "transport": "streamable_http",
    "description": "Hindsight memory"
  }'
```

This approach is useful when you want repeatable environment setup instead of manual admin clicks.

## Single-bank vs multi-bank mode

Hindsight's MCP endpoint supports two patterns.

### Multi-bank mode

Use the root MCP path:

```text
http://hindsight-api:8888/mcp
```

This exposes bank management tools and lets clients choose or create banks dynamically.

Use this when:

- many teams or workflows need different banks
- the gateway needs flexibility
- you want runtime bank selection

### Single-bank mode

Use a bank-pinned URL:

```text
http://hindsight-api:8888/mcp/my-team-bank/
```

This removes bank selection from the client side and pins all memory operations to one bank.

Use this when:

- one ContextForge team should map to one shared memory bank
- you want simpler client behavior
- you want tighter isolation by configuration

For deeper recall behavior, see [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall). For storage semantics, see [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Connect clients to ContextForge

Once Hindsight is registered, clients should connect to ContextForge's MCP endpoint instead of Hindsight directly.

For example, Claude Desktop might use:

```json
{
  "mcpServers": {
    "context-forge": {
      "url": "https://your-contextforge.com/mcp",
      "headers": {
        "Authorization": "Bearer <your-contextforge-token>"
      }
    }
  }
}
```

From the client's point of view, Hindsight is now just one capability behind the gateway.

## Verify the tools are present

After registration, list tools through ContextForge and confirm Hindsight appears.

```bash
curl -s -X POST https://your-contextforge.com/mcp \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}'
```

You should see Hindsight tools such as retain, recall, and reflect.

A practical test is:

1. store a fact with retain
2. recall it through the gateway
3. ask reflect for a synthesis over the saved memory

If all three work through ContextForge, the integration is doing what it should.

## Common mistakes

### Registering the wrong Hindsight URL

Make sure you point ContextForge at Hindsight's MCP path, not just the API root.

### Choosing multi-bank when you really want strict isolation

If a team should always operate in one bank, a bank-pinned URL is often simpler and safer.

### Testing Hindsight directly instead of through ContextForge

That proves Hindsight works, but not that the gateway registration is correct.

### Forgetting to verify permissions

If ContextForge auth is mis-scoped, the Hindsight server can be healthy while the client still cannot use the tools.

## FAQ

### Should every client connect to Hindsight directly?

Not if ContextForge is the pattern you want. The point of the gateway is to centralize the connection surface.

### Is Hindsight Cloud compatible with this pattern?

Yes. The key question is whether ContextForge can reach the Hindsight endpoint you register.

### When should I use a bank per team?

When memory should be shared inside a team but isolated from other teams.

### Is this better than adding Hindsight to each tool manually?

For teams with many tools, usually yes. For a single client, direct setup is often simpler.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want the easiest managed backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Compare cross-tool memory patterns in [Team Shared Memory for AI Coding Agents](https://hindsight.vectorize.io/blog/team-shared-memory-ai-coding-agents)
