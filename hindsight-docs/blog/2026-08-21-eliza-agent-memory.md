---
title: "Give Your elizaOS Agent a Long-Term Memory"
authors: [benfrank241]
slug: "2026/08/21/eliza-agent-memory"
date: 2026-08-21T12:00
tags: [hindsight, eliza, elizaos, agent-memory, autonomous-agents, multi-tenant]
description: "elizaOS agents run around the clock and talk to everyone. The Hindsight plugin gives them persistent, per-user long-term memory: recall before each model call, retain after each turn, fail-safe by default."
image: /img/blog/eliza-agent-memory.png
hide_table_of_contents: true
---

![Hindsight for elizaOS: a provider that recalls memories before each model call and an evaluator that retains conversations after each turn, with one memory bank per user](/img/blog/eliza-agent-memory.png)

[elizaOS](https://github.com/elizaOS/eliza) agents are built to run around the clock. You define a character, wire up a few plugins, and the agent lives in a Discord server or a Telegram group or on X, answering whoever talks to it, forever. Which surfaces a problem the coding-assistant crowd rarely hits at the same scale: an always-on agent talks to *everyone*, and without long-term memory it meets each of them as a stranger, every single time.

Today the [`@vectorize-io/hindsight-eliza`](https://www.npmjs.com/package/@vectorize-io/hindsight-eliza) plugin gives an elizaOS agent persistent memory backed by [Hindsight](https://hindsight.vectorize.io), and it does it with the two seams elizaOS already exposes: a provider and an evaluator.

<!-- truncate -->

## TL;DR

- **`@vectorize-io/hindsight-eliza`** adds long-term memory to any elizaOS agent. Add it to your character's `plugins` list and you are done.
- It registers two components: a **provider** (`HINDSIGHT_MEMORY`) that recalls relevant memories into the prompt before each model call, and an **evaluator** (`HINDSIGHT_RETAIN`) that retains the conversation after each turn.
- Both are **enabled by default**, layer on top of elizaOS's own memory, and **fail safe**: a memory-service hiccup never stops the agent from replying.
- Memory defaults to **one bank per user** (keyed by the message's `entityId`), so a public agent keeps every person's memory isolated without any extra wiring.
- Retain is **fire-and-forget by default**, so persistence adds no latency to a turn.

## Why memory is make-or-break for an always-on agent

A coding agent forgets your conventions and you re-explain them; annoying, but you are the only user. An elizaOS agent is a different shape. It is autonomous, it is long-lived, and it is usually talking to a crowd. Two things break without persistent, scoped memory:

- **It forgets across restarts.** Deploy a new version, or the process cycles overnight, and everything the agent learned about its community is gone. A social agent that cannot remember yesterday is not a personality, it is a goldfish with a good prompt.
- **It blurs users together, or leaks between them.** The whole point of a public agent is that it talks to many people. If their memory lands in one undivided pile, the agent either confuses them or, worse, recalls one person's context while talking to another. That is not a feature, it is an incident.

So the useful primitive is not just "memory," it is *persistent memory, scoped per user.* That is exactly what the plugin defaults to.

## How it plugs in: two seams, both by default

elizaOS composes each turn from **providers** and evaluates it afterward with **evaluators**, and the plugin hooks one of each:

| Component | elizaOS seam | When it runs | What it does |
| --- | --- | --- | --- |
| `HINDSIGHT_MEMORY` | Provider | During prompt composition, before the model call | Calls Hindsight `recall` with the incoming message and injects the relevant memories into context |
| `HINDSIGHT_RETAIN` | Evaluator | After the agent processes the turn | Calls Hindsight `retain` to persist the message, and optionally the agent's own replies |

Two design choices matter. First, both sides **fail safe**: if recall errors, the provider returns nothing and the turn proceeds on the base prompt; retain is wrapped so a write failure is swallowed rather than thrown. A memory outage degrades the agent to its normal, memoryless self instead of taking it down. Second, the plugin **layers on top of** elizaOS's existing short-term memory rather than replacing it, so you are adding durable recall, not swapping out the framework's context handling.

## The bank is the boundary

In Hindsight, memory lives in a **bank**, and a bank is a hard recall boundary: an agent can only recall what is in the bank it reads from. The plugin defaults the bank to the message's `entityId`, which means **every user gets their own isolated memory store automatically.** For a public agent, that default is the right one, and it is enforced by storage rather than by prompting.

When you need a different shape, `bank` takes either a fixed string or a function of the message:

```ts
createHindsightPlugin({
  client,
  bank: (message) => message.entityId, // the default: one bank per user
  // bank: "community-shared",          // or one shared brain for everyone
  // bank: (m) => m.roomId,             // or one bank per room/channel
});
```

One string gives the whole agent a single shared memory. A function lets you scope per user, per room, or per channel, whatever matches how your community is organized. The decision is a single field, and it is the whole privacy design.

## Tuning recall and retain

Both sides accept options, and the defaults are sensible, so tuning is optional:

```ts
createHindsightPlugin({
  client,
  recall: {
    budget: "high",        // "low" | "mid" | "high": latency vs. depth
    includeEntities: true, // fold in entity observations
    maxTokens: 1000,       // cap what gets injected into the prompt
  },
  retain: {
    async: true,           // fire-and-forget; never adds turn latency
    tags: ["source:eliza"],
    includeAgentMessages: false, // store user turns only, by default
  },
});
```

A couple of these are worth calling out for an autonomous agent. `retain.async` is `true` by default, so persistence happens off the critical path and never slows a reply. And `includeAgentMessages` is `false` by default, so the agent stores what its users say, not its own chatter, which keeps memory grounded in real signal instead of the model's own output. Flip it on when you want the agent to remember its own commitments too.

## Two setups worth copying

**A community agent that actually knows its members.** Keep the default per-user bank. Over a week, the agent learns that one person is a Rust developer asking about a specific bug, another runs the events channel, another always wants the short answer. None of it leaks between them, and none of it resets when you redeploy. The agent stops being a stateless bot and starts being a regular.

**A shared brain for a focused agent.** Point `bank` at a single fixed string and the agent remembers *collectively*: a support agent in one channel builds up a shared picture of the recurring issues, and every user benefits from what it learned talking to the last one. Use this when the memory is about the topic, not the person.

## Setup

If you already run an elizaOS agent on `@elizaos/core` `^1.7.2`, wiring this in is two steps:

```bash
npm install @vectorize-io/hindsight-eliza @vectorize-io/hindsight-client
```

```ts
import { createHindsightPlugin } from "@vectorize-io/hindsight-eliza";
import { Hindsight } from "@vectorize-io/hindsight-client";

export const character = {
  name: "Ada",
  plugins: [
    // ...your other plugins
    createHindsightPlugin({ client: new Hindsight({ apiKey: process.env.HINDSIGHT_API_KEY }) }),
  ],
};
```

That is the whole thing: both the recall provider and the retain evaluator are on the moment the agent runs. Memory can live in [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) or a server you host yourself, and because Hindsight is MIT licensed and self-hosts in one Docker command, a whole fleet of elizaOS agents can run with persistent memory entirely on your own hardware. For the full walkthrough and a verification flow, follow the [elizaOS memory guide](https://hindsight.vectorize.io/guides/2026/07/17/guide-eliza-memory-with-hindsight).

Give your always-on agent a memory that survives its restarts and keeps every user separate. Start free on [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup).

---

**Learn more:**
- [elizaOS integration docs](https://hindsight.vectorize.io/sdks/integrations/eliza) — the provider, the evaluator, and every option
- [Guide: Add elizaOS Memory with Hindsight](https://hindsight.vectorize.io/guides/2026/07/17/guide-eliza-memory-with-hindsight) — step-by-step setup and verification
- [One Bank or Many? Structuring Agent Memory](https://hindsight.vectorize.io/blog/2026/07/16/bank-strategy-agent-memory) — how to scope memory per user, per room, or shared
