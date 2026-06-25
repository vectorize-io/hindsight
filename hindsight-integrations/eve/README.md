# Hindsight for Eve

Long-term memory for [Vercel Eve](https://github.com/vercel/eve) agents, powered by
[Hindsight](https://vectorize.io/hindsight). One file gives your agent `retain`, `recall`,
and `reflect` over [Hindsight's MCP server](https://hindsight.vectorize.io) — so it
remembers facts across sessions and deployments instead of starting cold every time.

## How it works

Eve is filesystem-first: an agent gains a capability by dropping a file under
`agent/connections/`. This package wraps eve's `defineMcpClientConnection`, pre-filling the
Hindsight MCP endpoint, a model-facing description, and bearer auth. The model discovers the
tools through `connection__search` and calls them as `connection__hindsight__recall`,
`connection__hindsight__retain`, and `connection__hindsight__reflect`. The connection's URL
and token never reach the model.

## Install

```bash
npm install @vectorize-io/hindsight-eve
```

`eve` is a peer dependency — you already have it in an Eve project.

## Quick start

Create `agent/connections/hindsight.ts`:

```ts
import { defineHindsightConnection } from "@vectorize-io/hindsight-eve";

export default defineHindsightConnection();
```

That's it. By default the connection reads:

| Env var                 | Purpose                                                          |
| ----------------------- | ---------------------------------------------------------------- |
| `HINDSIGHT_API_KEY`     | Bearer token sent as `Authorization: Bearer <key>`               |
| `HINDSIGHT_MCP_URL`     | MCP endpoint (defaults to Hindsight Cloud)                       |
| `HINDSIGHT_MCP_BANK_ID` | Optional bank to scope memory to, sent as the `X-Bank-Id` header |

### Hindsight Cloud

Set `HINDSIGHT_API_KEY` to a key from your [Hindsight Cloud](https://hindsight.vectorize.io)
dashboard. The connection defaults to `https://api.hindsight.vectorize.io/mcp`, so no URL is
needed.

### Self-hosted

Point at your own server and (optionally) pick a bank:

```bash
export HINDSIGHT_MCP_URL="http://localhost:8000/mcp"
export HINDSIGHT_MCP_BANK_ID="my-project"
export HINDSIGHT_API_KEY="…"   # or omit and pass apiKey: null below for a no-auth server
```

```ts
import { defineHindsightConnection } from "@vectorize-io/hindsight-eve";

// A local server with no auth:
export default defineHindsightConnection({
  url: "http://localhost:8000/mcp",
  apiKey: null,
});
```

## Options

```ts
defineHindsightConnection({
  url, // string  — MCP endpoint; defaults to HINDSIGHT_MCP_URL, then Cloud
  apiKey, // string | null — bearer token; null = no auth (local dev)
  bankId, // string  — scope memory to a bank (X-Bank-Id header)
  description, // string  — override the model-facing description
  tools, // { allow } | { block } — narrow which Hindsight tools the model sees
  approval, // human-in-the-loop policy, e.g. once() from "eve/tools/approval"
});
```

Restrict the agent to read-only recall, and require approval the first time:

```ts
import { defineHindsightConnection } from "@vectorize-io/hindsight-eve";
import { once } from "eve/tools/approval";

export default defineHindsightConnection({
  tools: { allow: ["recall", "reflect"] },
  approval: once(),
});
```

## Verify

With the connection in place, run your agent and ask it something it would need to look up
("what did we decide about X last week?"). Eve's `connection__search` surfaces the Hindsight
tools and the model calls `connection__hindsight__recall`. To seed memory, have the agent
`retain` a fact in one session and `recall` it in the next.

## Links

- [Hindsight docs](https://hindsight.vectorize.io)
- [Eve connections](https://github.com/vercel/eve/blob/main/docs/connections.mdx)
