---
sidebar_position: 38
title: "Eve Agent Memory with Hindsight | Integration"
description: "Add automatic long-term memory to Vercel Eve agents with Hindsight. Memory is injected before each turn and retained after — no model tool-calling."
---

# Eve

Automatic long-term memory for [Vercel Eve](https://github.com/vercel/eve) agents using [Hindsight](https://vectorize.io/hindsight). Eve is filesystem-first — an agent gains a capability by dropping a file under `agent/`. The `@vectorize-io/hindsight-eve` package wires two files that call Hindsight's REST API directly, so your agent gets memory that **just works** — relevant memory is injected before every turn and each exchange is retained after — **without the model ever choosing to call a tool.**

## Install

```bash
npm install @vectorize-io/hindsight-eve
```

`eve` is a peer dependency you already have in an Eve project.

## Quick Start

Create two files:

```ts
// agent/instructions/hindsight.ts — recall: inject memory before each turn
import { hindsightMemory } from "@vectorize-io/hindsight-eve";

export default hindsightMemory();
```

```ts
// agent/hooks/hindsight.ts — retain: save each exchange after the turn
import { hindsightRetainHook } from "@vectorize-io/hindsight-eve";

export default hindsightRetainHook();
```

Both read their config from the environment:

| Env var             | Purpose                                                        |
| ------------------- | -------------------------------------------------------------- |
| `HINDSIGHT_API_KEY` | Bearer token sent as `Authorization: Bearer <key>`             |
| `HINDSIGHT_API_URL` | Hindsight REST base (defaults to Hindsight Cloud)              |
| `HINDSIGHT_BANK_ID` | Bank to scope memory to (defaults to `default`; auto-created)  |

### Hindsight Cloud

Set `HINDSIGHT_API_KEY` from your [Hindsight Cloud](https://hindsight.vectorize.io) dashboard. `HINDSIGHT_API_URL` defaults to `https://api.hindsight.vectorize.io`, so no URL is needed.

### Self-hosted

```ts
import { hindsightMemory } from "@vectorize-io/hindsight-eve";

// A local server with no auth:
export default hindsightMemory({ apiUrl: "http://localhost:8000", apiKey: null });
```

## Options

Both factories accept the same options (each falls back to its env var):

```ts
hindsightMemory({
  apiUrl,      // REST base; defaults to HINDSIGHT_API_URL, then Cloud
  apiKey,      // bearer token; null = no auth (local dev)
  bankId,      // bank to scope memory to
  recallQuery, // the broad query used for recall (see below)
  budget,      // "low" | "mid" | "high" — recall result budget (default "mid")
  maxTokens,   // recall token budget (default 1024)
  context,     // `context` tag written on retained items (default "eve")
  includeAssistantReply, // also retain the assistant's reply (default true)
  timeoutMs,   // HTTP timeout (default 15000)
  onError,     // (err, phase) => void — failures degrade silently (default console.warn)
});
```

## Recall is profile-based, not per-message

Eve's instruction resolver runs at the start of a turn and **cannot see the live user message**, so recall uses a fixed broad query (default: `"user preferences, identity, and working context"`) to surface the user's ambient profile/context each turn. This is ideal for "the agent knows you" — preferences, identity, ongoing context — and is fully deterministic. Tune it with `recallQuery`. Per-message, query-specific retrieval inherently needs a tool the model calls and is out of scope here.

## Verify

Run your agent. Tell it a durable preference in one chat ("whenever you write me code, use Python with full type hints and no comments"). Start a **fresh** chat and ask for something — the agent applies the remembered preference, because the memory was injected before the model ran, with no tool call.

## Links

- [Hindsight docs](https://hindsight.vectorize.io)
- [Eve hooks](https://github.com/vercel/eve/blob/main/docs/guides/hooks.md) · [Eve dynamic capabilities](https://github.com/vercel/eve/blob/main/docs/guides/dynamic-capabilities.md)
