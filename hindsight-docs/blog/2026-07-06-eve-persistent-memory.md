---
title: "Eve Persistent Memory: Automatic, No Tool Calls"
authors: [benfrank241]
slug: "2026/07/06/eve-persistent-memory"
date: 2026-07-06T12:00
tags: [hindsight, eve, vercel, integration, memory, persistent-memory, tutorial]
description: "Vercel Eve agents get automatic long-term memory with hindsight-eve v0.2.0: context injected before every turn, each exchange saved after, no model tool call."
image: /img/blog/eve-persistent-memory.png
hide_table_of_contents: true
---

![Eve Persistent Memory with Hindsight](/img/blog/eve-persistent-memory.png)

[Vercel Eve](https://vercel.com/eve) is an open-source, filesystem-first framework for building AI agents: an agent is a directory of files, a tool is one TypeScript file, and a skill is one Markdown file. Vercel [announced it](https://vercel.com/blog/introducing-eve) in June 2026, and it ships production features like durable execution, sandboxed compute, and OpenTelemetry tracing by default. What it does not ship is long-term memory, and that is where [Hindsight](https://hindsight.vectorize.io) comes in.

The first version of the Hindsight integration for [Eve](https://github.com/vercel/eve) gave the agent memory *tools*: `recall`, `retain`, and `reflect`, exposed over MCP for the model to call. That works, but it has a catch common to every tool-based memory setup. The model has to *decide* to use it. If it doesn't reach for `recall`, the memory may as well not exist.

`@vectorize-io/hindsight-eve` v0.2.0 removes that dependency. Memory is now **automatic**: relevant context is injected before every turn, and each exchange is saved after, without the model ever choosing to call a tool.

<!-- truncate -->

## TL;DR

- v0.2.0 switches from model-called memory tools to **automatic memory**.
- Two authored files: an **instructions resolver** that recalls before each turn and injects the result as a system message, and a **hook** that retains after each turn.
- Both call Hindsight's REST API directly, so memory never depends on the LLM deciding to act.
- Recall is **profile-based, not per-message** (the resolver can't see the live user message). It is ideal for "the agent knows you," and tunable.
- Works with [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server; retains run async and never block a turn.

## Why Automatic Beats a Memory Tool

A tool-based memory layer is only as reliable as the model's judgment about when to use it. Give an agent a `recall` tool and it will use it sometimes: when the prompt obviously calls for history, when it happens to think of it. The rest of the time it answers from a cold context and quietly forgets that you always want Python with type hints, or that this project standardised on a particular convention last week.

That unreliability is the whole problem [agent memory](https://vectorize.io/what-is-agent-memory) is supposed to solve. If remembering is optional, you are back to a stateless agent that occasionally remembers. Making recall and retain automatic, on every turn, is what turns "has a memory tool" into "actually remembers."

## How Eve's Automatic Memory Works

Eve is filesystem-first: an agent gains behaviour by dropping a file into the project. The v0.2.0 integration wires two of them, and neither exposes a tool to the model.

**`agent/instructions/hindsight.ts`** is a dynamic instructions resolver. On `turn.started`, before the model runs, it recalls from your Hindsight bank and injects the results as a system message. The block is fenced with a sentinel comment so recalled facts are never re-retained on the way back out.

**`agent/hooks/hindsight.ts`** is a hook. On `turn.completed`, it retains the user's message (and optionally the assistant's reply) to the bank. Retains run asynchronously, so they never add latency to a turn, and failures degrade quietly through an `onError` callback rather than breaking the agent.

Both files call Hindsight's REST API directly (recall via `POST /v1/default/banks/{bank}/memories/recall`, retain via `POST /v1/default/banks/{bank}/memories`). There is no MCP server and no tool for the model to call. Memory happens around the turn, not inside it.

![A retained preference consolidated into an observation in the Hindsight bank, no tool call involved.](/img/blog/eve-bank-observation.png)

## Install and Quick Start

```bash
npm install @vectorize-io/hindsight-eve
```

`eve` is a peer dependency, so you already have it in an Eve project. Then create two files:

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

That is the whole integration. Both read their config from the environment:

| Env var | Purpose |
| --- | --- |
| `HINDSIGHT_API_KEY` | Bearer token, sent as `Authorization: Bearer <key>` |
| `HINDSIGHT_API_URL` | Hindsight REST base (defaults to Hindsight Cloud) |
| `HINDSIGHT_BANK_ID` | Bank to scope memory to (defaults to `default`, auto-created) |

Point both files at the same `HINDSIGHT_BANK_ID` so recall and retain share one store (for example, one bank per user).

## Recall Is Profile-Based, Not Per-Message

This is the design trade-off worth understanding. Eve's instruction resolver runs at the *start* of a turn, before the live user message is available to it. So recall can't be query-specific to what the user just asked. Instead it uses a fixed, broad query (the default is `"user preferences, identity, and working context"`) to surface the user's ambient profile every turn.

That is exactly right for the "the agent knows you" case: preferences, identity, ongoing project context, the durable things that should shape every reply. It is deterministic, and you can tune the query with the `recallQuery` option to match what your agent should always keep in view.

What it is not is per-message semantic retrieval. Pulling the three most relevant facts for *this specific question* inherently needs a tool the model calls at the right moment, and that is out of scope for the automatic path by design. If you need both, the two approaches compose: automatic profile recall for the ambient context, a called tool for targeted lookups.

## Cloud or Self-Hosted

For **Hindsight Cloud**, set `HINDSIGHT_API_KEY` to a key from your dashboard. `HINDSIGHT_API_URL` defaults to `https://api.hindsight.vectorize.io`, so there is nothing else to configure.

For a **self-hosted** server, point at your own base URL. If it runs without auth, pass `apiKey: null`:

```ts
import { hindsightMemory } from "@vectorize-io/hindsight-eve";

export default hindsightMemory({ apiUrl: "http://localhost:8000", apiKey: null });
```

Both factories accept the same options, each falling back to its env var:

| Option | Default | Purpose |
| --- | --- | --- |
| `apiUrl` / `apiKey` / `bankId` | env, then Cloud | connection and bank (`apiKey: null` = no auth) |
| `recallQuery` | `"user preferences, identity, and working context"` | the broad query used for recall |
| `budget` | `"mid"` | recall result budget (`low` / `mid` / `high`) |
| `maxTokens` | `1024` | recall token budget |
| `includeAssistantReply` | `false` | also retain the assistant's reply (recommended: it often holds the solution) |
| `context` | `"eve"` | the `context` tag written on retained items |
| `onError` | `console.warn` | where recall/retain failures degrade to |

## Verify It Works

Run your agent and tell it a durable preference in one chat, for example "when I ask for code, always write it in Rust."

![Teaching the agent a durable preference in one chat.](/img/blog/eve-teach-preference.png)

Then start a **fresh** chat and ask for something. The agent applies the remembered preference, because the memory was injected as a system message before the model ran, with no tool call and no prompting from you.

![A fresh chat writes the CSV parser in Rust, applying the remembered preference with no tool call.](/img/blog/eve-recall-rust.png)

## Frequently Asked Questions

**Does the model have to call a tool to use memory?**
No. That is the point of v0.2.0. Recall is injected before the model runs, and retain happens after the turn via a hook. The model never sees or calls a memory tool.

**What gets recalled each turn?**
Your ambient profile: preferences, identity, and working context, via a fixed broad query. The instruction resolver runs before the live user message is available, so recall is profile-based rather than per-message. Tune it with `recallQuery`.

**Does it store the assistant's replies too?**
By default it retains only the user's message. But the assistant's reply is often where the real signal lives: the decision it reached, the solution it described, the code it wrote. For most coding and problem-solving agents you will want to set `includeAssistantReply: true` so both sides of the exchange are remembered, not just what the user asked.

**Does it add latency?**
Retains run asynchronously and never block a turn; recall is a single call before the turn. Failures degrade through `onError` rather than breaking the agent.

## Further reading

- [What is agent memory?](https://vectorize.io/what-is-agent-memory): the foundational concepts behind recall and retention.
- [Best AI agent memory systems](https://vectorize.io/articles/best-ai-agent-memory-systems): how the major agent memory frameworks compare.
- [Vercel AI SDK persistent memory](/blog/2026/06/23/vercel-ai-sdk-persistent-memory): memory for the other Vercel agent stack.
- [One memory for every AI tool](/blog/2026/04/07/one-memory-for-every-ai-tool): point Eve and your other agents at the same bank.
