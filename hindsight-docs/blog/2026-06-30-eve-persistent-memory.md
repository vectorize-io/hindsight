---
title: "Eve Persistent Memory: One File, Recall Across Sessions"
authors: [benfrank241]
slug: "2026/06/30/eve-persistent-memory"
date: 2026-06-30T12:00
tags: [hindsight, eve, vercel, integration, memory, agents, mcp, tutorial]
description: "Add persistent memory to Vercel Eve agents with one connection file: recall, retain, and reflect over Hindsight's MCP server, with a key the model never sees."
image: /img/blog/eve-persistent-memory.png
hide_table_of_contents: true
---

![Eve Persistent Memory with Hindsight](/img/blog/eve-persistent-memory.png)

[Vercel Eve](https://github.com/vercel/eve) is filesystem-first: an agent gains a capability by dropping a file under `agent/connections/`. Want a tool? Add a file. Want to change behavior? Edit a file. It's a clean model, and it stops at the session boundary. Every run starts cold, with no idea what the agent decided, learned, or was told last time.

The `@vectorize-io/hindsight-eve` connection closes that gap in the most Eve-native way possible: with one file. Drop it in and your agent gets `recall`, `retain`, and `reflect` over [Hindsight's](https://vectorize.io/hindsight) [Model Context Protocol](https://modelcontextprotocol.io) (MCP) server, so it carries [agent memory](https://vectorize.io/what-is-agent-memory) across sessions and deployments instead of relearning everything each time.

<!-- truncate -->

## TL;DR

- Vercel Eve agents are stateless between runs; nothing survives the session boundary by default.
- `@vectorize-io/hindsight-eve` adds memory as a single connection file, `agent/connections/hindsight.ts`.
- The model gets three tools over MCP: `recall` (search), `retain` (store), `reflect` (synthesized answer), discovered through Eve's `connection__search`.
- The connection holds the endpoint URL and bearer token. **They never reach the model.**
- Works with [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server, and supports least-privilege controls: restrict tools and require human approval.

---

## Why Eve Agents Need Persistent Memory

An Eve agent can read its workspace and whatever files you've given it. What it can't do is remember anything from the last time it ran. The convention you established on Monday, the customer detail it learned mid-conversation, the architectural call it talked through last week: all gone the moment the session ends.

You can hard-code facts into a file, and for stable knowledge that works. It doesn't work for the things the agent discovers as it goes, the details you didn't know to write down in advance. That's the gap a memory layer fills: the agent retains durable facts as it works, and recalls the relevant ones automatically next time.

For Eve specifically, this matters because of how the framework is used. Eve agents are long-lived operators, support handlers, coding assistants, ops bots, the kind of agent you talk to repeatedly. An operator that reintroduces itself every session isn't much of an operator. Persistent memory is what turns a stateless script into something that compounds.

Picture a `release-notes` agent you run after every deploy. The first time, you tell it your team writes notes in past tense and groups changes by surface area. Without memory, you tell it again next Friday, and the Friday after that. With a Hindsight connection it retains the convention once and recalls it on every future run, so the notes come out in your house style without the reminder. The agent didn't get smarter; it stopped forgetting.

## How Eve Persistent Memory Works

The integration leans entirely into Eve's filesystem-first design. A capability is a file, so memory is a file too.

Under the hood, `defineHindsightConnection` wraps Eve's `defineMcpClientConnection`, pre-filling three things: the Hindsight MCP endpoint, a model-facing description of what the memory tools do, and bearer authentication. The agent discovers the tools at runtime through Eve's `connection__search`, then calls them as `connection__hindsight__recall`, `connection__hindsight__retain`, and `connection__hindsight__reflect`.

The detail worth pausing on: **the connection's URL and token never reach the model.** Eve resolves them on the host side when it makes the MCP call. The model sees tool names and a description, never your Hindsight key. For an agent that might log its own context or get prompt-injected, that separation is the difference between a leaked credential and a non-event.

## Install and Quick Start

```bash
npm install @vectorize-io/hindsight-eve
```

`eve` is a peer dependency, so you already have it in an Eve project (the connection targets `eve >= 0.11.0`).

Now create `agent/connections/hindsight.ts`:

```ts
import { defineHindsightConnection } from "@vectorize-io/hindsight-eve";

export default defineHindsightConnection();
```

That's the whole integration. By default the connection reads three environment variables:

| Env var | Purpose |
| --- | --- |
| `HINDSIGHT_API_KEY` | Bearer token, sent as `Authorization: Bearer <key>` |
| `HINDSIGHT_MCP_URL` | MCP endpoint (defaults to Hindsight Cloud) |
| `HINDSIGHT_MCP_BANK_ID` | Optional bank to scope memory to, sent as the `X-Bank-Id` header |

## Cloud or Self-Hosted

For **Hindsight Cloud**, set `HINDSIGHT_API_KEY` to a key from your dashboard and you're done. The connection defaults to `https://api.hindsight.vectorize.io/mcp`, so there's no URL to configure.

For a **self-hosted** server, point at your own endpoint and optionally pick a bank. If your local server runs without auth, pass `apiKey: null` and no `Authorization` header is sent:

```ts
import { defineHindsightConnection } from "@vectorize-io/hindsight-eve";

export default defineHindsightConnection({
  url: "http://localhost:8000/mcp",
  apiKey: null,
});
```

A `bankId` scopes memory so different agents or projects don't read each other's history. Leave it unset to share one bank, or set it per agent for isolation.

## Least Privilege: Restrict Tools and Require Approval

Memory tools are powerful, and Eve lets you fence them in. `defineHindsightConnection` takes a `tools` allow/block list and an `approval` policy, so you decide exactly what the model can do.

A common shape is a read-only agent that can pull context but never write to memory, with a one-time human approval the first time it reaches for the tools:

```ts
import { defineHindsightConnection } from "@vectorize-io/hindsight-eve";
import { once } from "eve/tools/approval";

export default defineHindsightConnection({
  tools: { allow: ["recall", "reflect"] },
  approval: once(),
});
```

Now the agent can `recall` and `reflect` but not `retain`, and the first invocation pauses for a human to approve. For a customer-facing or autonomous agent, that's the difference between "reads memory" and "rewrites memory unsupervised." You can also override the model-facing `description` to steer when the agent decides to reach for memory at all.

## Verify It Works

With the connection file in place, run your agent and ask it something it would have to look up: "what did we decide about the billing migration last week?" Eve's `connection__search` surfaces the Hindsight tools, and the model calls `connection__hindsight__recall`. To prove the round trip, have the agent `retain` a fact in one session, restart, and ask for it in the next. If it comes back, memory is live across the session boundary, which was the whole point.

## Frequently Asked Questions

**Does Vercel Eve have built-in memory across sessions?**
No. Eve agents are stateless between runs. Persistent memory comes from a connection like `@vectorize-io/hindsight-eve` that gives the agent recall and retain over a memory layer.

**Does the model ever see my Hindsight API key?**
No. The connection holds the endpoint URL and bearer token, and Eve attaches them host-side when it calls the MCP server. The model only sees the tool names and their description.

**Can I stop the agent from writing to memory?**
Yes. Pass `tools: { allow: ["recall", "reflect"] }` to expose read-only memory, and add an `approval` policy to require human sign-off on tool use.

**Does it work with a self-hosted Hindsight?**
Yes. Set `url` (or `HINDSIGHT_MCP_URL`) to your server. For an open local server with no auth, pass `apiKey: null` and no `Authorization` header is sent.

## Further reading

- [What is agent memory?](https://vectorize.io/what-is-agent-memory): the foundational concepts behind recall, retention, and memory banks.
- [Best AI agent memory systems](https://vectorize.io/articles/best-ai-agent-memory-systems): how the major agent memory frameworks compare.
- [Vercel AI SDK persistent memory](/blog/2026/06/23/vercel-ai-sdk-persistent-memory): memory for the other Vercel agent stack.
- [One memory for every AI tool](/blog/2026/04/07/one-memory-for-every-ai-tool): point Eve and your other agents at the same bank.
