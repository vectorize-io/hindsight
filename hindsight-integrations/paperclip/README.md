# @vectorize-io/hindsight-paperclip

Persistent memory for [Paperclip AI](https://github.com/paperclipai/paperclip) agents using [Hindsight](https://hindsight.vectorize.io).

Paperclip agents start every heartbeat cold — no memory of prior sessions, decisions, or patterns. This package gives them long-term memory that persists across heartbeats and sessions.

## How It Works

1. **Before each heartbeat**: `recall()` queries Hindsight for context relevant to the current task and injects it into the agent's prompt
2. **After each heartbeat**: `retain()` stores the agent's output so future heartbeats can reference it

Memory is isolated per company and agent by default (`paperclip::{companyId}::{agentId}`), matching Paperclip's multi-tenant model.

## Installation

```bash
npm install @vectorize-io/hindsight-paperclip
```

## Configuration

Set environment variables (or pass as options to `loadConfig()`):

| Variable | Description | Default |
|---|---|---|
| `HINDSIGHT_API_URL` | Hindsight server URL | Required |
| `HINDSIGHT_API_TOKEN` | API token for Hindsight Cloud | — |

## Usage

### HTTP Adapter Agents (Express middleware)

```typescript
import express from 'express'
import { createMemoryMiddleware, loadConfig } from '@vectorize-io/hindsight-paperclip'
import type { HindsightRequest } from '@vectorize-io/hindsight-paperclip'

const app = express()
app.use(express.json())
app.use(createMemoryMiddleware(loadConfig()))

app.post('/heartbeat', async (req, res) => {
  const { memories, runId } = (req as HindsightRequest).hindsight
  const { context } = req.body

  const prompt = memories
    ? `Past context:\n${memories}\n\nCurrent task: ${context.taskDescription}`
    : `Task: ${context.taskDescription}`

  const output = await runYourAgent(prompt)
  res.json({ output })  // middleware auto-retains output
})
```

The middleware reads `agentId`, `companyId`, `runId`, and `context.taskDescription` from the Paperclip HTTP adapter request body automatically.

### Process Adapter Scripts

```typescript
import { recall, retain, loadConfig } from '@vectorize-io/hindsight-paperclip'

const config = loadConfig()
const { PAPERCLIP_AGENT_ID, PAPERCLIP_COMPANY_ID, PAPERCLIP_RUN_ID } = process.env

// Recall before executing
const memories = await recall({
  agentId: PAPERCLIP_AGENT_ID!,
  companyId: PAPERCLIP_COMPANY_ID!,
  query: process.env.TASK_DESCRIPTION ?? '',
}, config)

if (memories) {
  console.log(`[Memory Context]\n${memories}`)
}

// ... agent does its work ...

// Retain after
await retain({
  agentId: PAPERCLIP_AGENT_ID!,
  companyId: PAPERCLIP_COMPANY_ID!,
  content: agentOutput,
  documentId: PAPERCLIP_RUN_ID!,
}, config)
```

### Direct Function Usage

```typescript
import { recall, retain, loadConfig } from '@vectorize-io/hindsight-paperclip'

const config = loadConfig({
  hindsightApiUrl: 'https://api.hindsight.vectorize.io',
  hindsightApiToken: process.env.HINDSIGHT_API_TOKEN,
})

const memories = await recall(
  { companyId, agentId, query: `${task.title}\n${task.description}` },
  config
)

if (memories) {
  systemPrompt = `Past context:\n${memories}\n\n${systemPrompt}`
}
```

## Bank ID Isolation

By default, each company+agent pair gets its own memory bank:

```
paperclip::{companyId}::{agentId}
```

You can change the isolation granularity:

```typescript
// Shared memory across all agents in a company
loadConfig({ bankGranularity: ['company'] })
// → "paperclip::{companyId}"

// Agent's global memory across all companies
loadConfig({ bankGranularity: ['agent'] })
// → "paperclip::{agentId}"

// Custom prefix
loadConfig({ bankIdPrefix: 'myapp' })
// → "myapp::{companyId}::{agentId}"
```

## Configuration Reference

```typescript
interface PaperclipMemoryConfig {
  hindsightApiUrl: string          // HINDSIGHT_API_URL — required
  hindsightApiToken?: string       // HINDSIGHT_API_TOKEN
  bankGranularity?: ('company' | 'agent')[]  // default: ['company', 'agent']
  bankIdPrefix?: string            // default: 'paperclip'
  recallBudget?: 'low' | 'mid' | 'high'      // default: 'mid'
  recallMaxTokens?: number         // default: 1024
  retainContext?: string           // default: 'paperclip'
  timeoutMs?: number               // default: 15000
}
```

## Skill File

An agent-readable skill file is included at `src/skills/hindsight.md`. Inject it into your agent's system prompt or as a Paperclip skill to give the agent direct access to Hindsight's REST API via `curl`.

## Requirements

- Node.js 20+ (uses native `fetch`)
- Hindsight server (self-hosted or [Hindsight Cloud](https://hindsight.vectorize.io))
