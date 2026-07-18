---
title: "Guide: Add Flowise Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, flowise, agents, memory]
description: "Add Flowise memory with Hindsight using three Tool nodes — Retain, Recall, Reflect — so any chatflow or agent can store, recall, and reason over long-term memory."
image: /img/guides/guide-flowise-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add Flowise Memory with Hindsight](/img/guides/guide-flowise-memory-with-hindsight.svg)

If you want **Flowise memory with Hindsight**, the setup is three Tool nodes — **Hindsight Retain**, **Hindsight Recall**, and **Hindsight Reflect** — that drop into any chatflow or agent flow alongside your other LangChain tools. Each node returns a LangChain `DynamicStructuredTool`, so it slots into any agent that accepts tools, and together they give a flow long-term memory across sessions instead of staying stateless.

This is a good fit for Flowise because Flowise is the LangChain-derived visual builder, and its agents already know how to call tools. You attach the memory nodes, share one Hindsight credential across them, and the agent learns to recall context before answering and retain it after meaningful exchanges. Memory is scoped per bank, so you can isolate one user or session from another by setting a Default Bank ID on each node.

This guide walks through copying the nodes into a Flowise checkout, creating the Hindsight credential, wiring the three tools into a conversational agent, and a quick verification flow so you can confirm memory is actually being used. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. Copy the Hindsight Retain, Recall, and Reflect nodes plus the credential class into a Flowise checkout, then `pnpm add @vectorize-io/hindsight-client`, `pnpm install`, `pnpm build`, `pnpm start`.
> 2. Create a **Hindsight API** credential with your API URL and `hsk_...` key.
> 3. Attach the three tool nodes to a Conversational Agent, all sharing that credential.
> 4. Set a Default Bank ID like `user-${sessionId}` on each node so memory is per user.
> 5. Verify that a later turn recalls what an earlier one stored.

## Prerequisites

Before you start, make sure you have:

- A Flowise checkout you can build and run locally (`git clone` of `FlowiseAI/Flowise`)
- `pnpm` available to build and run Flowise
- A reachable Hindsight backend, [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server
- A Hindsight API key (`hsk_...`) from the Hindsight dashboard

## Step 1: Copy the nodes into a Flowise checkout

Flowise distributes nodes only inside its main monorepo, so installation today means using a Flowise checkout with the Hindsight nodes copied in. The user-facing distribution will be the upstream PR to `FlowiseAI/Flowise`.

```bash
git clone https://github.com/FlowiseAI/Flowise.git
cd Flowise

# Copy the three tool nodes
cp -r /path/to/hindsight/hindsight-integrations/flowise/nodes/tools/Hindsight* \
  packages/components/nodes/tools/

# Copy the credential class
cp /path/to/hindsight/hindsight-integrations/flowise/credentials/HindsightApi.credential.ts \
  packages/components/credentials/

# Add the client dep
cd packages/components && pnpm add @vectorize-io/hindsight-client
cd ../.. && pnpm install && pnpm build
pnpm start  # opens http://localhost:3000
```

Once Flowise starts, the three Hindsight tool nodes are available in the node palette alongside your other tools.

## Step 2: Create the Hindsight credential

In Flowise, create a new credential of type **Hindsight API**:

- **API URL** — defaults to `https://api.hindsight.vectorize.io` (Cloud); change it for a self-hosted server
- **API Key** — your `hsk_...` key (optional for self-hosted unauthenticated instances)

If you don't have a key yet, [sign up free at Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) and grab an API key from the dashboard, or [self-host](https://hindsight.vectorize.io/developer/installation) a server. All three tool nodes can share the same credential.

## Step 3: Attach the three tools to an agent

Each Hindsight tool node returns a LangChain `DynamicStructuredTool`, so it slots into any Flowise agent that accepts tools. A typical conversational support agent recipe:

1. **ChatOpenAI** (or any chat LLM)
2. A **Conversational Agent** with three tools attached: Hindsight Retain + Hindsight Recall + Hindsight Reflect, all sharing the same Hindsight API credential
3. Set the **Default Bank ID** to something like `user-${sessionId}` on each tool
4. The agent learns to call Recall before answering and Retain after meaningful exchanges. Use Reflect for "what do we know about this user?" prompts.

## What each tool does

The three nodes map to Hindsight's core memory operations, and the agent passes the input fields at call time:

- **Hindsight Retain** stores free-text content in a memory bank; Hindsight extracts facts asynchronously after the call returns. The node has a **Default Bank ID** field, and the agent passes `bankId`, `content`, and optional `tags`.
- **Hindsight Recall** searches a bank for memories relevant to a query and returns ranked results. It has **Default Bank ID** and **Default Budget** (`low` / `mid` / `high`) fields, and the agent passes `bankId`, `query`, and optional `budget`, `maxTokens`, `tags`.
- **Hindsight Reflect** returns an LLM-synthesized answer over the bank. It has **Default Bank ID** and **Default Budget** fields, and the agent passes `bankId`, `query`, and optional `budget`.

The Default Bank ID on each node is used when the agent doesn't pass its own `bankId`. For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Verify that memory is working

A good test sequence is:

1. run a chatflow with all three Hindsight tools attached and a Default Bank ID set
2. tell the agent something it should remember (a preference, a decision, a fact)
3. let the agent call Retain on that exchange
4. start a new turn or session against the same bank
5. ask about the earlier detail so the agent calls Recall

For example:

- turn one tells the agent the user prefers concise answers in Rust
- a later turn asks the agent how it should format code for this user

If the agent surfaces the earlier preference through Recall, the setup is working. You can also use Reflect to ask a synthesizing question like "what do we know about this user?"

## Common mistakes

### Nodes not sharing a credential

All three tool nodes should point at the same Hindsight API credential. If one uses a different credential or URL, it may write to or read from a different backend.

### Bank IDs that don't line up

Retain and Recall have to target the same bank for memory to carry over. If Retain writes to one Default Bank ID and Recall reads from another, the earlier context won't surface.

### Testing recall before retain runs

Hindsight extracts facts asynchronously after Retain returns. If you recall immediately, the newest content may not be fully processed yet.

### Expecting memory without attaching the tools

Flowise chatflows are stateless across sessions on their own. The memory only exists once the Hindsight tool nodes are attached and the agent actually calls them.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — set the credential's API URL to your self-hosted URL. The API Key is optional for self-hosted unauthenticated instances.

### How is memory scoped?

Per bank. Set a Default Bank ID on each node — something like `user-${sessionId}` isolates memory per user or session.

### What's the difference between Recall and Reflect?

Recall searches the bank and returns ranked results for the agent to read. Reflect returns a single LLM-synthesized answer over the bank, which is better for questions like "what do we know about this customer?"

### Why copy the nodes instead of installing a package?

Flowise distributes nodes only inside its main monorepo, so today you copy the nodes into a Flowise checkout. The user-facing distribution will be an upstream PR to `FlowiseAI/Flowise`.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [Flowise integration docs](https://hindsight.vectorize.io/docs/integrations/flowise)
