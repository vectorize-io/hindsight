---
title: "Guide: Add elizaOS Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, eliza, agents, memory]
description: "Add elizaOS memory with Hindsight using the @vectorize-io/hindsight-eliza plugin, so your agent recalls relevant memories before each model call and retains conversations after each turn."
image: /img/guides/guide-eliza-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add elizaOS Memory with Hindsight](/img/guides/guide-eliza-memory-with-hindsight.svg)

If you want **elizaOS memory with Hindsight**, the cleanest setup is the `@vectorize-io/hindsight-eliza` plugin. You add it to your character's plugin list, and it registers two components on the agent: a provider that recalls relevant memories into the prompt before each model call, and an evaluator that retains conversation messages after each turn. That gives your elizaOS agent long-term memory across conversations instead of forgetting everything between turns.

This is a good fit for elizaOS because the framework already exposes provider and evaluator seams. The plugin hooks into both: `HINDSIGHT_MEMORY` runs during prompt composition to inject recalled memory, and `HINDSIGHT_RETAIN` runs after the turn to persist what was said. Both are enabled by default, layer on top of elizaOS's existing memory, and fail safe — a Hindsight outage never blocks the agent from responding.

This guide walks through installing the plugin, wiring in a Hindsight client, understanding the per-user bank strategy, and a quick verification flow so you can confirm memory is actually being used. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. `npm install @vectorize-io/hindsight-eliza @vectorize-io/hindsight-client`.
> 2. Create a `Hindsight` client with your `HINDSIGHT_API_KEY`.
> 3. Build the plugin with `createHindsightPlugin({ client, ... })`.
> 4. Add the plugin to your character's `plugins` list — memory defaults to one bank per user.
> 5. Verify that a later turn remembers what an earlier one stored.

## Prerequisites

Before you start, make sure you have:

- An elizaOS agent using `@elizaos/core` `^1.7.2` (the plugin declares it as a peer dependency)
- A reachable Hindsight backend, [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server
- A Hindsight API key or the base URL for your self-hosted server, available to the client

## Step 1: Install the plugin

Install the plugin alongside the Hindsight client.

```bash
npm install @vectorize-io/hindsight-eliza @vectorize-io/hindsight-client
```

The package targets `@elizaos/core` `^1.7.2`, declared as a peer dependency, so install it into an existing elizaOS project.

## Step 2: Create the plugin and add it to your character

Create the plugin with a Hindsight client and add it to your character's `plugins` list:

```ts
import { createHindsightPlugin } from "@vectorize-io/hindsight-eliza";
import { Hindsight } from "@vectorize-io/hindsight-client";

const hindsightPlugin = createHindsightPlugin({
  client: new Hindsight({ apiKey: process.env.HINDSIGHT_API_KEY }),
  recall: { budget: "high", includeEntities: true },
  retain: { tags: ["source:eliza"] },
});

export const character = {
  name: "Ada",
  plugins: [
    // ...your other plugins
    hindsightPlugin,
  ],
};
```

That is the whole setup. Both the recall provider and the retain evaluator are enabled by default, so the agent starts recalling and retaining as soon as it runs.

## Step 3: Tune recall and retain (optional)

The plugin accepts options for both sides. These are the documented options and their defaults:

```ts
createHindsightPlugin({
  client,

  // Which memory bank to read/write. A string uses one fixed bank for all
  // messages; a function derives the bank per message. Defaults to
  // `message.entityId` (one bank per user).
  bank: (message) => message.entityId,

  recall: {
    enabled: true,            // set false to disable recall
    budget: "mid",            // "low" | "mid" | "high" — latency vs. depth
    types: ["world", "experience"], // restrict to fact types
    maxTokens: 1000,          // cap recalled tokens
    includeEntities: false,   // include entity observations
    heading: "# Relevant long-term memories", // prompt heading
  },

  retain: {
    enabled: true,            // set false to disable retain
    async: true,              // fire-and-forget; never adds turn latency
    tags: ["source:eliza"],   // tags on every retained memory
    metadata: { env: "prod" },
    includeAgentMessages: false, // also store the agent's own replies
  },
});
```

To run only one side, set `recall.enabled: false` or `retain.enabled: false`. You can also build the components directly with `createHindsightProvider` and `createHindsightEvaluator` if you want to wire them into a plugin yourself.

## How the plugin uses memory

elizaOS exposes provider and evaluator seams, and the plugin uses both:

- **Recall (before):** the `HINDSIGHT_MEMORY` provider runs during prompt composition, before the model call. It calls Hindsight `recall` with the incoming message and injects the results into context under the configured heading.
- **Retain (after):** the `HINDSIGHT_RETAIN` evaluator runs after the agent processes the turn. It calls Hindsight `retain` to persist the message — and optionally the agent's own replies when `includeAgentMessages` is set.

With `retain.async` on by default, retain is fire-and-forget, so it never adds latency to a turn.

For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Per-user memory banks

By default each agent message is stored under a memory **bank** keyed by the message's `entityId`, giving every user an isolated memory store. That keeps one user's memory from leaking into another's.

If you want a different strategy, pass `bank` as a fixed string to use one shared bank for all messages, or a `(message) => string` function to derive the bank per message. See the [integration docs](https://hindsight.vectorize.io/docs/integrations/eliza) for the full option list.

## Verify that memory is working

A good test sequence is:

1. run your agent with the plugin installed
2. in one turn, tell the agent something specific to remember
3. let the turn finish so the message is retained
4. in a later turn, ask the agent about what you told it earlier

For example:

- turn one tells the agent your preferred programming language
- a later turn asks the agent what language you prefer

If the recalled memory surfaces the earlier detail, the setup is working.

## Common mistakes

### Forgetting the client dependency

The plugin needs a Hindsight client, so install and import `@vectorize-io/hindsight-client` and pass an instance as `client`.

### Not adding the plugin to the character

`createHindsightPlugin(...)` only builds the plugin. It has no effect until you add it to your character's `plugins` list.

### Testing retain too early

The message is retained after the turn is processed. If you check for it mid-turn, it may not have been stored yet — and with `retain.async` on, retain runs in the background.

### Assuming memory is shared across users

By default each user gets their own bank keyed by `entityId`. That is usually what you want, but do not expect one user to recall another user's context unless you set a fixed `bank`.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — point your Hindsight client at your server instead of Hindsight Cloud.

### Can I run only recall or only retain?

Yes. Set `recall.enabled: false` or `retain.enabled: false`. You can also assemble your own plugin with `createHindsightProvider` and `createHindsightEvaluator`.

### How is memory scoped?

Per user by default, using the message `entityId` as the bank. Pass a `bank` string or `(message) => string` function to change this.

### Does it slow down my agent?

No. Both components fail safe, and retain is fire-and-forget by default (`retain.async`), so a memory-service hiccup never blocks or delays a response.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [elizaOS integration docs](https://hindsight.vectorize.io/docs/integrations/eliza)
