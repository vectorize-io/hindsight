# Hindsight for Eve

Automatic long-term memory for [Vercel Eve](https://github.com/vercel/eve) agents, powered by
[Hindsight](https://vectorize.io/hindsight). Two files give your agent memory that **just
works** — relevant memory is injected before every turn, and each exchange is saved after —
**without the model ever choosing to call a tool.**

## How it works

Eve is filesystem-first. This package wires two authored files that call Hindsight's REST API
directly, so memory never depends on the LLM deciding to call a tool:

- **`agent/instructions/hindsight.ts`** — a dynamic instructions resolver that, before each
  turn, recalls the user's stored memory from Hindsight and injects it as a system message.
- **`agent/hooks/hindsight.ts`** — a hook that, after each turn, retains the user message and
  the assistant's answer to Hindsight.

## Install

```bash
npm install @vectorize-io/hindsight-eve
```

`eve` is a peer dependency — you already have it in an Eve project.

## Quick start

Create two files:

```ts
// agent/instructions/hindsight.ts
import { hindsightMemory } from "@vectorize-io/hindsight-eve";

export default hindsightMemory();
```

```ts
// agent/hooks/hindsight.ts
import { hindsightRetainHook } from "@vectorize-io/hindsight-eve";

export default hindsightRetainHook();
```

That's it. Both read their config from the environment:

| Env var             | Purpose                                                       |
| ------------------- | ------------------------------------------------------------- |
| `HINDSIGHT_API_KEY` | Bearer token sent as `Authorization: Bearer <key>`            |
| `HINDSIGHT_API_URL` | Hindsight REST base (defaults to Hindsight Cloud)             |
| `HINDSIGHT_BANK_ID` | Bank to scope memory to (defaults to `default`; auto-created) |

### Hindsight Cloud

Set `HINDSIGHT_API_KEY` to a key from your [Hindsight Cloud](https://hindsight.vectorize.io)
dashboard. `HINDSIGHT_API_URL` defaults to `https://api.hindsight.vectorize.io`, so no URL is
needed.

### Self-hosted

```bash
export HINDSIGHT_API_URL="http://localhost:8000"
export HINDSIGHT_BANK_ID="my-project"
export HINDSIGHT_API_KEY="…"   # or pass apiKey: null below for a no-auth server
```

```ts
import { hindsightMemory } from "@vectorize-io/hindsight-eve";

export default hindsightMemory({ apiUrl: "http://localhost:8000", apiKey: null });
```

## Options

Both factories accept the same options (each falls back to its env var):

```ts
hindsightMemory({
  apiUrl, // string  — REST base; defaults to HINDSIGHT_API_URL, then Cloud
  apiKey, // string | null — bearer token; null = no auth (local dev)
  bankId, // string  — bank to scope memory to
  recallQuery, // string  — the broad query used for recall (see below)
  budget, // "low" | "mid" | "high" — recall result budget (default "mid")
  maxTokens, // number  — recall token budget (default 1024)
  context, // string  — `context` tag written on retained items (default "eve")
  includeAssistantReply, // boolean — also retain the assistant's reply (default false)
  timeoutMs, // number  — HTTP timeout (default 15000)
  onError, // (err, phase) => void — failures degrade silently via this (default console.warn)
});
```

## Recall is profile-based, not per-message

Eve's instruction resolver runs at the start of a turn and **cannot see the live user
message**, so recall uses a fixed broad query (default:
`"user preferences, identity, and working context"`) to surface the user's ambient
profile/context each turn. This is ideal for "the agent knows you" — preferences, identity,
ongoing context — and is deterministic. Tune it with `recallQuery`. (Per-message, query-
specific retrieval inherently needs a tool the model calls; that's out of scope here.)

## Notes

- Memory is scoped to a **bank** (one isolated store, e.g. per user). Point both files at the
  same `HINDSIGHT_BANK_ID`.
- By default only the **user's** message is retained (the durable signal) — set
  `includeAssistantReply: true` to also store the assistant's reply.
- Retains run asynchronously and never block a turn; failures degrade via `onError`.
- The recall block injected into context is fenced with a sentinel so recalled facts are never
  re-retained.

## Verify

Run your agent. Tell it a durable preference in one chat ("whenever you write me code, use
Python with full type hints and no comments"). Start a **fresh** chat and ask for something —
the agent applies the remembered preference, because the memory was injected before the model
ran, with no tool call.

## Links

- [Hindsight docs](https://hindsight.vectorize.io)
- [Eve hooks](https://github.com/vercel/eve/blob/main/docs/guides/hooks.md) ·
  [Eve dynamic capabilities](https://github.com/vercel/eve/blob/main/docs/guides/dynamic-capabilities.md)
