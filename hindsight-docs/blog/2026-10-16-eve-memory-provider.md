---
title: "Four Things to Build With Vercel Eve and Hindsight"
authors: [benfrank241]
slug: "2026/10/16/eve-memory-provider"
date: 2026-10-16T12:00
tags: [hindsight, eve, vercel, integration, memory, agent-memory, multi-tenant, use-cases]
description: "A team glossary that outlives the session, one customer across every channel, tenant isolation that is not a filter, and the memory your other AI tools already wrote."
image: /img/blog/eve-memory-provider.png
hide_table_of_contents: true
---

![Hindsight as a memory provider for a Vercel Eve memory slot](/img/blog/eve-memory-provider.png)

Tell a [Vercel Eve](https://eve.dev) agent that revenue is net of refunds and a week starts Monday, and for the rest of that conversation it gets your numbers right. Tomorrow it asks again. The colleague sitting next to you never gets the benefit at all.

Eve gives memory a first-class home — a slot, a scope, and a provider — and [Hindsight](https://vectorize.io/hindsight) is one you can drop into it. Here are four things that get easier once an agent's memory outlives the session, and what each one takes.

<!-- truncate -->

## TL;DR

- **A glossary that outlives the session.** Scope a memory slot to the team instead of the conversation and house definitions stop being re-taught.
- **One customer, every channel.** Memory follows the authenticated principal, so Slack in March and WhatsApp in October are the same person — as long as one identity provider says so.
- **Tenant isolation that is not a filter.** Each locked scope gets its own Hindsight bank, so separation is a property of where you look rather than a `WHERE` clause you remembered.
- **The memory your stack already wrote.** Point the slot at a bank your coding agents already fill, and the Eve agent starts knowing what they learned.

## 1. A glossary that outlives the session

Eve's own tutorial has the analytics agent learn exactly this kind of vocabulary, and stores it with `defineState`, which their docs describe as surviving across step and turn boundaries *within a session*. Right instinct, wrong lifetime. The glossary is not a fact about this conversation; it is a fact about the team.

Re-explaining it every session is a waste, and having each colleague teach the agent privately is worse, because what the agent knows then depends on who it is talking to.

Move the glossary into a memory slot scoped to the team and both problems go at once:

```ts
// agent/memory/glossary.ts
import { defineMemory } from "eve/memory";
import { hindsightMemory } from "@vectorize-io/hindsight-eve";

export default defineMemory({
  description: "The team's house definitions for metrics and analysis.",
  provider: hindsightMemory(),
  scope: (ctx) => {
    const team = ctx.session.auth.current?.attributes.teamId;
    return typeof team === "string" ? [team] : null;
  },
});
```

Now "revenue is net of refunds" is learned once by whoever says it first, and recalled for everyone after that. Nobody calls a tool to save it and nobody calls a tool to look it up: Hindsight captures the exchange after the turn and recalls what is relevant to the next question before the model runs.

Returning `null` from a scope resolver is worth knowing about. Eve disables memory for that turn rather than falling back to a shared scope, so an unauthenticated caller gets an agent with no memory instead of somebody else's glossary.

## 2. One customer, every channel

Eve ships channels for Slack, Discord, Teams, Telegram, Twilio, GitHub, Linear and the chat SDKs. A support agent usually ends up on two or three of them, and the awkward part is not the plumbing. It is that a customer who explained their setup over WhatsApp in March opens the web widget in October and explains it again.

Scope the slot by the person rather than the conversation and that stops:

```ts
import { byPrincipal } from "eve/memory/scope";

export default defineMemory({
  description: "What this customer has told us, across every channel.",
  provider: hindsightMemory(),
  scope: byPrincipal,
});
```

**One caveat worth stating plainly,** because it decides whether this works for you. `byPrincipal` builds its key from the principal type, the authenticator, the issuer and the principal ID. Two channels share memory when they resolve to the same authenticated principal. If each channel authenticates people through its own system, the same human arrives as two principals and gets two banks.

That is the correct default — guessing that a Slack ID and a phone number are the same person is how memory leaks between strangers. But if you have a real identity layer, scope on the stable user ID it gives you and the continuity follows the human instead of the channel.

## 3. Tenant isolation that is not a filter

Every multi-tenant agent has the same quiet risk: isolation implemented as a condition that one query forgot. It is fine until the day it isn't, and the failure is silent, because a filter that does not match simply returns less.

Eve's position is that this is a scope decision rather than a storage one. Resolve the tenant from verified session auth, never from the model, and return a tuple:

```ts
scope(ctx) {
  const caller = ctx.session.auth.current;
  const tenantId = caller?.attributes.tenantId;
  const principal = byPrincipal(ctx);
  if (caller?.principalType !== "user" || typeof tenantId !== "string" || principal === null) {
    return null;
  }
  return [tenantId, principal];
},
visibility: "scope",
```

Eve derives an opaque key from that tuple, locks it before any recall runs, and hands the same key to every read, write and tool. The model never sees it and cannot change it. Provider tools close over the locked scope, so a tool cannot be talked into pointing at another tenant.

Hindsight maps each locked scope to **its own bank**. That matters more than it sounds for a semantic store: a nearest-neighbour search that ranks globally and filters afterwards has already read the other tenant's vectors, and it returns fewer results than you asked for in a way that looks like a relevance problem rather than a leak. Searching one tenant's bank is a different operation from searching everyone's and discarding.

Keep `visibility: "scope"`, which is the default. If the authenticated caller changes inside one session, Eve hides the earlier scope's recalled records rather than leaving them in view.

## 4. The memory your stack already wrote

The first three are things any good provider should do. This one is specific to Hindsight, and it is the reason an Eve agent can start out already knowing things.

A Hindsight bank is not owned by the agent that writes it. If your engineers run Hindsight behind their coding agents, there is already a bank that knows why the retry logic looks the way it does and which migration broke staging in August:

```bash
npx @vectorize-io/hindsight-coding-agents install all
```

Point an Eve slot at that bank and the agent inherits all of it:

```ts
provider: hindsightMemory({ bankId: "engineering-shared" }),
```

An internal agent on Slack answering "why does checkout retry three times" can draw on what the team worked out in their editors, rather than on whatever made it into a doc. The same applies in reverse: what the Eve agent learns is in a bank the other tools can read.

Be deliberate here. A string `bankId` pins **every** scope to one bank, which is exactly right for a shared internal agent and exactly wrong for a customer-facing one, because it switches off the isolation from the previous section. For per-user memory that also reaches a shared bank, keep the scoped default for the slot and give the shared bank its own slot. Eve supports several, and each gets its own scope.

## What it takes

One file, for all four:

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

```bash
npm install @vectorize-io/hindsight-eve
```

Set `HINDSIGHT_API_KEY` and the slot is live: recall on the live user message before every turn, capture after it, a `reflect` tool when a question needs reasoning across memories rather than retrieval of a few. No model tool-calling on the path that matters, and a recall that fails degrades to an agent without memory rather than a failed turn.

Eve 0.69.0 or later is worth insisting on. Versions 0.51.0 through 0.68.x never ran `capture["turn.completed"]` for any provider, so memory was recalled and nothing new was stored.

## Further reading

- [Hindsight for Eve](/sdks/integrations/eve): install, options, and how scopes map to banks.
- [Eve memory](https://eve.dev/docs/memory) and [multi-tenant memory](https://github.com/vercel/eve/blob/main/docs/patterns/multi-tenant-memory.md): the slot, scope and visibility model these examples use.
- [One memory for every AI tool](/blog/2026/04/07/one-memory-for-every-ai-tool): the shared-bank idea behind the fourth use case.
- [What is agent memory?](https://vectorize.io/what-is-agent-memory): what ends up in the store, and why it is not a transcript.
