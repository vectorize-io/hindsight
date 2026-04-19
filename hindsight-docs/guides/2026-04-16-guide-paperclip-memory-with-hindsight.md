---
title: "Guide: Add Paperclip Memory with Hindsight"
authors: [benfrank241]
date: 2026-04-16
tags: [how-to, paperclip, agents, memory]
description: "Add Paperclip memory with Hindsight so agents can retain, recall, and reflect across heartbeats and sessions instead of starting cold each run."
image: /img/blog/guide-paperclip-memory-with-hindsight.png
hide_table_of_contents: true
---

![Guide: Add Paperclip Memory with Hindsight](/img/blog/guide-paperclip-memory-with-hindsight.png)

If you want **Paperclip memory with Hindsight**, the key move is to wrap the Paperclip heartbeat loop with Hindsight recall before execution and Hindsight retain after execution. That gives a Paperclip agent durable memory across heartbeats, instead of forcing every run to start cold.

This is useful because heartbeat-based agents often repeat the same expensive context gathering over and over. They rediscover the same user preferences, retry the same failed approaches, and lose the outcome of prior runs. Hindsight gives Paperclip a long-term memory layer so the next heartbeat can build on what the last one learned.

This guide covers the quick-start path, the HTTP and process adapter patterns, bank isolation by company and agent, and the checks that tell you the memory layer is really working. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) open while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. Install `@vectorize-io/hindsight-paperclip`.
> 2. Load the config and call recall before the heartbeat executes.
> 3. Call retain after the agent produces output.
> 4. Keep bank IDs scoped by company and agent unless you intentionally want shared memory.
> 5. Verify that a later heartbeat sees what an earlier heartbeat stored.

## Prerequisites

Before you start, make sure you have:

- A Paperclip agent already running through the heartbeat model
- A reachable Hindsight backend, either self-hosted or [Hindsight Cloud](https://hindsight.vectorize.io)
- Stable identifiers for `companyId` and `agentId`

The identifier design matters. Paperclip's default isolation pattern works well because it maps cleanly to a multi-tenant setup.

## Step 1: Install the integration

Install the Paperclip package:

```bash
npm install @vectorize-io/hindsight-paperclip
```

Then load configuration from your environment.

```typescript
import { recall, retain, loadConfig } from '@vectorize-io/hindsight-paperclip'

const config = loadConfig()
```

The loader reads your Hindsight URL and token configuration, so you do not have to wire every request by hand.

## Step 2: Recall memory before each heartbeat

Before the heartbeat runs, query Hindsight for prior context.

```typescript
const memories = await recall({
  companyId,
  agentId,
  query: `${task.title}\n${task.description}`,
}, config)

if (memories) {
  systemPrompt = `Past context:\n${memories}\n\n${systemPrompt}`
}
```

This is the crucial step. The agent does not become stateful because it is running frequently. It becomes stateful because you inject relevant history before the new run starts.

## Step 3: Retain output after the heartbeat

Once the heartbeat finishes, store what the agent learned or produced.

```typescript
await retain({
  companyId,
  agentId,
  content: agentOutput,
  documentId: runId,
}, config)
```

That creates the feedback loop. One heartbeat produces context for the next heartbeat.

## How the default bank isolation works

Paperclip's default bank format is built around company and agent IDs:

```text
paperclip::{companyId}::{agentId}
```

That is a good default because it prevents unrelated tenants and agents from leaking into one another.

Common alternatives:

- **company-only** when several agents in one company should share the same memory
- **agent-only** when one agent should carry memory across several company contexts
- **custom prefix** when you want your own bank namespace scheme

If you want to reason more deeply about retrieval behavior later, review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall). For what gets stored and how, review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## HTTP adapter integration

If your Paperclip agent runs as an HTTP webhook server, use the middleware path.

```typescript
import express from 'express'
import { createMemoryMiddleware, loadConfig } from '@vectorize-io/hindsight-paperclip'
import type { HindsightRequest } from '@vectorize-io/hindsight-paperclip'

const app = express()
app.use(express.json())
app.use(createMemoryMiddleware(loadConfig()))

app.post('/heartbeat', async (req, res) => {
  const { memories } = (req as HindsightRequest).hindsight
  const { context } = req.body

  const prompt = memories
    ? `Past context:\n${memories}\n\nCurrent task: ${context.taskDescription}`
    : `Task: ${context.taskDescription}`

  const output = await runYourAgent(prompt)
  res.json({ output })
})
```

This is a good option when Paperclip is already operating as a service and you want memory to be part of the request lifecycle.

## Process adapter integration

If your Paperclip agent runs as a script, use the process-style recall and retain flow.

```typescript
import { recall, retain, loadConfig } from '@vectorize-io/hindsight-paperclip'

const config = loadConfig()
const { PAPERCLIP_AGENT_ID, PAPERCLIP_COMPANY_ID, PAPERCLIP_RUN_ID } = process.env

const memories = await recall({
  agentId: PAPERCLIP_AGENT_ID!,
  companyId: PAPERCLIP_COMPANY_ID!,
  query: process.env.TASK_DESCRIPTION ?? '',
}, config)

if (memories) {
  console.log(`[Memory Context]\n${memories}`)
}

await retain({
  agentId: PAPERCLIP_AGENT_ID!,
  companyId: PAPERCLIP_COMPANY_ID!,
  content: agentOutput,
  documentId: PAPERCLIP_RUN_ID!,
}, config)
```

This is the better fit when the heartbeat is invoked through a process adapter rather than an HTTP request path.

## Verify that memory is working

A simple verification loop is:

1. heartbeat one stores a fact or lesson
2. heartbeat two asks about the same issue
3. the second run sees relevant context without you re-encoding it manually

For example:

- first run learns that a certain customer always wants Slack alerts before email
- second run sees that preference already in memory and uses it automatically

That is the practical behavior you want to confirm.

## Common mistakes

### Treating logs as memory

Raw logs are not the same thing as useful memory. The point is to store meaningful, reusable context.

### Choosing the wrong bank granularity

If you accidentally scope memory too broadly, tenants or agents can bleed together.

### Forgetting the post-run retain step

Recall without retain gives you stale memory. Retain without recall gives you unused memory.

### Testing only one run

You need at least two runs to see whether memory actually compounds.

## FAQ

### Does Paperclip need Hindsight Cloud?

No. Self-hosted Hindsight works too.

### Should I share memory across all agents in a company?

Only if that is the intended behavior. Company-level sharing is powerful, but it should be a deliberate choice.

### Is middleware the only integration option?

No. The package supports both HTTP and process adapter patterns.

### Can I use this for team knowledge, not just user preferences?

Yes. Procedure outcomes, recurring failures, and operational knowledge are all strong fits.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Compare shared knowledge patterns in [Team Shared Memory for AI Coding Agents](https://hindsight.vectorize.io/blog/team-shared-memory-ai-coding-agents)
