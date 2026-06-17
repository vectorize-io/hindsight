---
sidebar_position: 38
title: "Eve Agent Memory with Hindsight | Integration"
description: "Add long-term memory to Vercel Eve agents with Hindsight. A one-line MCP connection gives your agent retain, recall, and reflect across sessions."
---

# Eve

Long-term memory for [Vercel Eve](https://github.com/vercel/eve) agents using [Hindsight](https://vectorize.io/hindsight). Eve is filesystem-first — an agent gains a capability by dropping a file under `agent/connections/`. The `@vectorize-io/hindsight-eve` package wraps Eve's `defineMcpClientConnection`, so one file gives your agent `retain`, `recall`, and `reflect` over Hindsight's MCP server and it remembers across sessions and deployments.

## Install

```bash
npm install @vectorize-io/hindsight-eve
```

`eve` is a peer dependency you already have in an Eve project.

## Quick Start

Create `agent/connections/hindsight.ts`:

```ts
import { defineHindsightConnection } from "@vectorize-io/hindsight-eve";

export default defineHindsightConnection();
```

The connection reads its defaults from the environment:

| Env var                 | Purpose                                                          |
| ----------------------- | ---------------------------------------------------------------- |
| `HINDSIGHT_API_KEY`     | Bearer token sent as `Authorization: Bearer <key>`               |
| `HINDSIGHT_MCP_URL`     | MCP endpoint (defaults to Hindsight Cloud)                       |
| `HINDSIGHT_MCP_BANK_ID` | Optional bank to scope memory to, sent as the `X-Bank-Id` header |

The model discovers the tools via Eve's `connection__search` and calls them as `connection__hindsight__recall`, `connection__hindsight__retain`, and `connection__hindsight__reflect`. The connection's URL and token never reach the model.

### Hindsight Cloud

Set `HINDSIGHT_API_KEY` from your [Hindsight Cloud](https://hindsight.vectorize.io) dashboard. The connection defaults to `https://api.hindsight.vectorize.io/mcp`, so no URL is needed.

### Self-hosted

Point at your own server, optionally scoping to a bank. Use `apiKey: null` for a no-auth local server:

```ts
import { defineHindsightConnection } from "@vectorize-io/hindsight-eve";

export default defineHindsightConnection({
  url: "http://localhost:8000/mcp",
  apiKey: null,
});
```

## Options

```ts
defineHindsightConnection({
  url,         // MCP endpoint; defaults to HINDSIGHT_MCP_URL, then Cloud
  apiKey,      // bearer token; null = no auth (local dev)
  bankId,      // scope memory to a bank (X-Bank-Id header)
  description, // override the model-facing description
  tools,       // { allow } | { block } — narrow which Hindsight tools the model sees
  approval,    // human-in-the-loop policy, e.g. once() from "eve/tools/approval"
});
```

Restrict the agent to read-only recall and require approval the first time:

```ts
import { defineHindsightConnection } from "@vectorize-io/hindsight-eve";
import { once } from "eve/tools/approval";

export default defineHindsightConnection({
  tools: { allow: ["recall", "reflect"] },
  approval: once(),
});
```

## Links

- [Hindsight docs](https://hindsight.vectorize.io)
- [Eve connections](https://github.com/vercel/eve/blob/main/docs/connections.mdx)
