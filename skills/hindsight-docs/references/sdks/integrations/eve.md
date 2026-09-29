
# Eve

Long-term memory for [Vercel Eve](https://eve.dev) agents using [Hindsight](https://vectorize.io/hindsight). The `@vectorize-io/hindsight-eve` package is a provider for Eve's [memory slots](https://eve.dev/docs/memory): relevant memory is recalled on the live user message before every turn, each exchange is captured after, and every scope (user, tenant, channel) gets its own isolated Hindsight bank — **without the model ever choosing to call a tool.**

[View Changelog →](../../changelog/integrations/eve.md)

## How it works

Eve resolves and locks a scope for every turn, then calls the provider at fixed points:

| Phase            | What Hindsight does                                                                                |
| ---------------- | -------------------------------------------------------------------------------------------------- |
| `turn.started`   | Recalls memories relevant to the user's message and places them in context, attributed to the slot |
| `turn.completed` | Retains the user message and the assistant's final reply (idempotent per Eve operation id)         |
| tools            | Exposes `<slot>__reflect`, so the model can ask long-term memory a question when it needs to       |

Every read and write is partitioned by the locked scope key: one Hindsight bank per scope, auto-created on first use. Two tenants can never see each other's memory.

## Install

```bash
npm install @vectorize-io/hindsight-eve
```

`eve` (>= 0.45.1) is a peer dependency you already have in an Eve project.

## Quick Start

Create one file:

```ts
// agent/memory/hindsight.ts
import { hindsightMemory } from "@vectorize-io/hindsight-eve";
import { defineMemory } from "eve/memory";
import { byPrincipal } from "eve/memory/scope";

export default defineMemory({
  description: "Long-term memory about the current user.",
  provider: hindsightMemory(),
  scope: byPrincipal,
});
```

The provider reads its connection from the environment:

| Env var             | Purpose                                                                  |
| ------------------- | ------------------------------------------------------------------------ |
| `HINDSIGHT_API_KEY` | Bearer token sent as `Authorization: Bearer <key>`                       |
| `HINDSIGHT_API_URL` | Hindsight REST base (defaults to Hindsight Cloud)                        |
| `HINDSIGHT_BANK_ID` | _Optional._ Pin every scope to one shared bank (single-user agents only) |

### Hindsight Cloud

Set `HINDSIGHT_API_KEY` from your [Hindsight Cloud](https://hindsight.vectorize.io) dashboard. `HINDSIGHT_API_URL` defaults to `https://api.hindsight.vectorize.io`, so no URL is needed.

### Self-hosted

```ts
// A local server with no auth:
provider: hindsightMemory({ apiUrl: "http://localhost:8000", apiKey: null }),
```

## Scope → bank

`scope` is Eve's — `byPrincipal` gives each authenticated caller their own memory; a custom resolver can scope by tenant, channel, or anything else in trusted session metadata (see [Eve's memory docs](https://eve.dev/docs/memory)). Hindsight maps each locked scope to a bank:

- **Default:** the bank is the scope key (`memscope1_…`), an opaque digest Eve derives from the namespace and scope. Isolation is automatic.
- **Custom:** pass a resolver to name banks yourself, e.g. ``bankId: (scope) => `eve-${scope.value}` `` for human-readable bank names in the Hindsight dashboard.
- **Shared:** a string `bankId` (or `HINDSIGHT_BANK_ID`) pins every scope to a single bank. Only do this for single-user agents — it disables per-scope isolation.

## Options

```ts
hindsightMemory({
  apiUrl, // REST base; defaults to HINDSIGHT_API_URL, then Cloud
  apiKey, // bearer token; null = no auth (local dev)
  bankId, // string | (scope) => string — see "Scope → bank"
  budget, // "low" | "mid" | "high" — recall result budget (default "mid")
  maxTokens, // recall token budget (default 1024)
  recallQuery, // query when the turn has no user text (default: a broad profile query)
  context, // `context` tag written on retained items (default "eve")
  includeAssistantReply, // also retain the assistant's reply (default true)
  capture, // retain each turn (default true; false = recall-only)
  tools, // expose the `reflect` tool (default true)
  timeoutMs, // HTTP timeout (default 15000)
  onError, // (err, phase) => void — failures degrade silently (default console.warn)
});
```

## Notes

- Recall never fails a turn: a Hindsight error is reported through `onError` and the turn runs without memory. Captures run asynchronously and never block a reply.
- Each turn's recall is one message with a stable id, so it **supersedes** the previous turn's block instead of piling up in session history — including after compaction.
- Captures are keyed by Eve's `operationId`; a replayed capture replaces the earlier document rather than storing it twice.

## Known issue: capture on eve 0.51.0 – 0.66.x

eve 0.51.0 stopped passing the turn's history to `turn.completed` (a regression from
[vercel/eve#2690](https://github.com/vercel/eve/pull/2690)), so **no provider's
`capture["turn.completed"]` runs** on those versions — memory is recalled but nothing new is
stored. Tracked in [vercel/eve#3223](https://github.com/vercel/eve/issues/3223) with a fix in
[vercel/eve#3465](https://github.com/vercel/eve/pull/3465). Recall, the `reflect` tool, and
compaction capture are unaffected. Until the fix ships, either pin `eve@0.50.0`, or keep the
deprecated `hindsightRetainHook()` in `agent/hooks/hindsight.ts` next to the provider — eve's hook
events still fire — giving both the same string `bankId` so they share a bank.

## Verify

Run your agent. Tell it a durable preference in one session ("whenever you write me code, use Python with full type hints and no comments"). Start a **fresh** session as the same user and ask for something — the agent applies the remembered preference, because the memory was recalled before the model ran, with no tool call.

## Migrating from 0.2

0.2 wired memory through two authored files (`agent/instructions/hindsight.ts` with `hindsightMemory()` and `agent/hooks/hindsight.ts` with `hindsightRetainHook()`), before Eve had a memory provider contract. Replace both with the single `agent/memory/hindsight.ts` above. The old entrypoints still ship as `hindsightAutoRecall()` and `hindsightRetainHook()` (deprecated, removed in the next release); `hindsightMemory()` now returns the provider.

What changes: recall uses the live user message instead of a fixed broad query, memory is isolated per scope instead of one bank for everyone, and recalled context arrives as a slot-attributed message rather than system instructions.

## Links

- [Hindsight docs](https://hindsight.vectorize.io)
- [Eve memory](https://eve.dev/docs/memory) · [Build a memory provider](https://github.com/vercel/eve/blob/main/docs/memory/custom-provider.md)
