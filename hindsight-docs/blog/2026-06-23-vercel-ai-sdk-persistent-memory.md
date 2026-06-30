---
title: "Persistent Memory for the Vercel AI SDK in Five Tools"
authors: [benfrank241]
slug: "2026/06/23/vercel-ai-sdk-persistent-memory"
date: 2026-06-23T12:00
tags: [vercel-ai-sdk, memory, persistent-memory, hindsight, agents, typescript, tutorial]
description: "Add long-term memory to any Vercel AI SDK app with Hindsight. Five ready-to-use tools for retain, recall, and reflect that work with generateText, streamText, and ToolLoopAgent, on any model provider."
image: /img/blog/vercel-ai-sdk-persistent-memory.png
hide_table_of_contents: true
---

![Persistent Memory for the Vercel AI SDK with Hindsight](/img/blog/vercel-ai-sdk-persistent-memory.png)

The [Vercel AI SDK](https://ai-sdk.dev) is the fastest way to ship an AI feature in TypeScript: one `generateText` call, swap model providers with a one-line import, stream tokens to a React component. What it doesn't give you is memory. Every call starts from an empty slate, so your assistant forgets the user's name, their preferences, and everything decided in the last conversation the moment the request ends.

The Hindsight integration closes that gap the way the AI SDK expects you to extend it: with **tools**. One `npm install` and the model gains five memory tools it can call to retain facts, recall context, and reflect over what it knows. It works with any model provider the SDK supports, because it lives entirely on the tool layer.

## TL;DR

<!-- truncate -->

- The Vercel AI SDK has no persistent memory. Each `generateText` / `streamText` call is stateless.
- `@vectorize-io/hindsight-ai-sdk` adds memory as **five AI SDK tools**: `retain`, `recall`, `reflect`, `getMentalModel`, and `getDocument`.
- One call wires them in: `createHindsightTools({ client, bankId })`, then pass the result to `tools`.
- The design splits responsibility cleanly: the **agent** controls semantic inputs (what to remember, what to search for); your **application** locks infrastructure (the bank, cost budget, tags, async mode). The model can't change the bank ID or blow your token budget.
- Works with `generateText`, `streamText`, and `ToolLoopAgent`, on any provider (OpenAI, Anthropic, Google, and the rest).
- Hindsight Cloud means no infrastructure to run. [Sign up free.](https://ui.hindsight.vectorize.io/signup)

## Why the AI SDK Needs Memory

The AI SDK gives you a clean request-shaped abstraction: messages in, text or a stream out, with tool calls in between. That statelessness is exactly what makes it easy to deploy on serverless and edge runtimes. It's also what makes memory your problem.

For anything you run more than once against the same user, the gap shows up fast. A support assistant re-asks for the account details every session. A coding helper re-learns the project's conventions on every reload. A personal assistant forgets the preferences the user told it yesterday. The usual fix is to stuff prior turns into the prompt, but that only stretches as far as the context window and it resets the moment the session ends. You either build a real memory layer yourself (a datastore, embeddings, retrieval, deduplication) or you ship an assistant with amnesia.

Hindsight is that memory layer, exposed as tools the AI SDK already knows how to drive.

## Memory as Tools, Done Right

The AI SDK is a tool-calling framework. The native way to give a model a new capability is to hand it a tool, so that's how this integration works: the model decides when to reach for memory, the same way it decides when to call a weather API or run a calculation.

The trap with tool-based memory is handing the model too many knobs. If the model gets to pick the bank ID, the cost budget, and the tagging strategy on every call, you've turned infrastructure decisions over to a language model, and it will get them wrong. Hindsight's integration avoids that by drawing a hard line between two kinds of inputs:

- **Semantic inputs belong to the agent.** What to remember, what to search for, what question to reflect on. These are language decisions, which is exactly what the model is good at.
- **Infrastructure belongs to your application.** Which bank to write to, how much latency to spend, what tags to attach, whether writes are fire-and-forget. These are fixed when you create the tools, and the model never sees them.

In practice that means the bank ID is set once, at construction, and the model literally cannot change it. The result is multi-user isolation you can trust and a token budget the model can't accidentally blow.

## The Five Tools

`createHindsightTools` registers five tools. The middle column is what the model fills in on each call; the right column is what you lock at construction time.

| Tool | Agent provides | Application controls |
| --- | --- | --- |
| `retain` | `content`, `documentId`, `timestamp`, `context` | `async`, `tags`, `metadata` |
| `recall` | `query`, `queryTimestamp` | `budget`, `types`, `maxTokens`, `includeEntities`, `includeChunks` |
| `reflect` | `query`, `context` | `budget`, `maxTokens` |
| `getMentalModel` | `mentalModelId` | — |
| `getDocument` | `documentId` | — |

The first three are the core loop. `retain` stores something worth remembering. `recall` searches memory for relevant facts. `reflect` reasons over those memories to synthesize an answer rather than just returning matches. The last two are for exact retrieval: `getMentalModel` pulls a consolidated, pre-synthesized summary that's cheaper than searching raw memories, and `getDocument` fetches a stored document by ID for cases where you need the exact original text back.

## Setup

Install the package alongside the AI SDK and the Hindsight client:

```bash
npm install @vectorize-io/hindsight-ai-sdk @vectorize-io/hindsight-client ai
```

The recommended backend is **Hindsight Cloud**: [sign up free](https://ui.hindsight.vectorize.io/signup), create an API key, and point the client at it. Self-hosting works identically; run the API locally with one command and use its URL instead:

```bash
uvx hindsight-embed@latest -p myapp daemon start
# API available at http://localhost:8000
```

## Adding Memory to a Generation

This is the whole integration. Create a client, create the tools with a `bankId`, and pass them to any AI SDK call:

```typescript
import { HindsightClient } from "@vectorize-io/hindsight-client";
import { createHindsightTools } from "@vectorize-io/hindsight-ai-sdk";
import { generateText } from "ai";
import { openai } from "@ai-sdk/openai";

const client = new HindsightClient({ baseUrl: process.env.HINDSIGHT_API_URL! });

const tools = createHindsightTools({
  client,
  bankId: "user-123",
});

const { text } = await generateText({
  model: openai("gpt-4o"),
  tools,
  maxSteps: 5,
  system: "You are a helpful assistant with long-term memory.",
  prompt: "Remember that I prefer dark mode and large fonts.",
});
```

On a later call, even from a cold serverless invocation, the model calls `recall` and gets the preferences back:

```typescript
const { text } = await generateText({
  model: openai("gpt-4o"),
  tools,
  maxSteps: 5,
  system: "You are a helpful assistant with long-term memory.",
  prompt: "What are my display preferences?",
});
// -> answers with dark mode and large fonts
```

Nothing in your request handler changed except adding `tools`. Setting `maxSteps` (or a `stopWhen` condition) is what lets the model call a memory tool and then use the result in its answer within a single turn.

## Streaming and Agents

Because memory is just tools, it drops into every AI SDK entry point unchanged.

With `streamText`, the tools work exactly the same; tokens stream while the model calls memory mid-generation:

```typescript
import { streamText } from "ai";

const result = streamText({
  model: openai("gpt-4o"),
  tools,
  maxSteps: 5,
  system: "You are a helpful assistant with long-term memory.",
  prompt: "What are my display preferences?",
});

for await (const chunk of result.textStream) {
  process.stdout.write(chunk);
}
```

With `ToolLoopAgent`, you hand the same tools to a self-driving agent loop:

```typescript
import { ToolLoopAgent, stepCountIs } from "ai";

const agent = new ToolLoopAgent({
  model: openai("gpt-4o"),
  tools: createHindsightTools({ client, bankId: "user-123" }),
  stopWhen: stepCountIs(10),
  system: "You are a helpful assistant with long-term memory.",
});

const result = await agent.generate({
  prompt: "Remember that my favorite editor is Neovim",
});
```

## Per-User Memory in a Next.js Route

The `bankId` is the routing key for memory, and a Hindsight bank is just a namespace. The most common pattern is one bank per user. Because the bank is fixed at construction, the clean way to do this on the server is to create the tools **inside** your request handler, closing over the current user's ID:

```typescript
// app/api/chat/route.ts
import { streamText } from "ai";
import { openai } from "@ai-sdk/openai";
import { HindsightClient } from "@vectorize-io/hindsight-client";
import { createHindsightTools } from "@vectorize-io/hindsight-ai-sdk";

const hindsightClient = new HindsightClient({
  baseUrl: process.env.HINDSIGHT_API_URL!,
});

export async function POST(req: Request) {
  const { messages, userId } = await req.json();

  // Tools are created per request, closing over the current user's bankId
  const tools = createHindsightTools({
    client: hindsightClient,
    bankId: userId,
  });

  return streamText({
    model: openai("gpt-4o"),
    tools,
    maxSteps: 5,
    system: "You are a helpful assistant with long-term memory.",
    messages,
  }).toDataStreamResponse();
}
```

Each request gets tools scoped to exactly one user's memory. There's no shared mutable state and no way for one user's request to read or write another's bank, because the model never controls which bank it's talking to.

## Tuning Without Touching Your Agent Loop

Every infrastructure concern is an option on `createHindsightTools`, grouped under the tool it affects. You set these once; the model never sees them:

```typescript
const tools = createHindsightTools({
  client,
  bankId: userId,

  retain: {
    async: true,                        // fire-and-forget (default: false)
    tags: ["env:prod", "app:support"],  // attached to every retained memory
    metadata: { version: "2.0" },
  },

  recall: {
    budget: "high",                     // low | mid | high (default: 'mid')
    types: ["experience", "world"],     // restrict fact types (default: all)
    maxTokens: 2048,                    // cap the token budget
    includeEntities: true,              // include entity observations
  },

  reflect: {
    budget: "mid",
  },
});
```

A few that matter in production:

- **`retain.async: true`** makes writes fire-and-forget, so a retain call doesn't add ingestion latency to the user's turn.
- **`recall.budget`** trades latency for depth: `low` for snappy lookups, `high` when thoroughness matters more than speed.
- **`recall.types`** restricts recall to `world`, `experience`, or `observation` facts when you only want one kind.
- **`retain.tags`** stamps every memory from this surface with a tag, which keeps a shared bank organized across multiple apps.

You can also override any tool's `description` to steer when the model reaches for it, without changing your system prompt.

## Why This Design Holds Up

| | AI SDK default | With Hindsight |
| --- | --- | --- |
| Memory across calls | None | Persistent, per bank |
| Setup | n/a | `createHindsightTools`, pass to `tools` |
| Mechanism | n/a | Five AI SDK tools |
| Model provider | Any | Any (tools are provider-agnostic) |
| Who controls the bank ID | n/a | The application, fixed at construction |
| Who controls cost/latency | n/a | The application, not the model |
| Multi-user isolation | n/a | Per-request `bankId` |

The split between semantic and infrastructure inputs is the part worth copying even if you build your own memory tools. Let the model decide what to remember and what to look up. Don't let it decide where memory lives or how much it costs. That's the difference between a memory layer you can run in production and a demo that works until a model picks the wrong bank.

## Next Steps

- **Hindsight Cloud:** [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io/signup)
- **Integration docs:** [Vercel AI SDK + Hindsight](/sdks/integrations/ai-sdk)
- **Source:** [vectorize-io/hindsight/hindsight-integrations/ai-sdk](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/ai-sdk)
- **Reflect, explained:** [Mental models and hierarchical retrieval](https://hindsight.vectorize.io/blog/2026/06/05/mental-models-deep-dive)
