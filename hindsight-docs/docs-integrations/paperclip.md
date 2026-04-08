---
sidebar_position: 11
title: "Paperclip Persistent Memory with Hindsight | Integration Guide"
description: "Add long-term memory to Paperclip agents with Hindsight. Retain, recall, and reflect memories across sessions using the Paperclip integration."
---

# Paperclip

Persistent memory for [Paperclip AI](https://github.com/paperclipai/paperclip) agents using [Hindsight](https://hindsight.vectorize.io).

Paperclip agents start every heartbeat cold — no memory of prior sessions, decisions, or patterns. The `@vectorize-io/hindsight-paperclip` package gives them long-term memory that persists across heartbeats and sessions.

## Quick Start

```bash
npm install @vectorize-io/hindsight-paperclip
```

```typescript
import { recall, retain, loadConfig } from '@vectorize-io/hindsight-paperclip'

const config = loadConfig()  // reads HINDSIGHT_API_URL, HINDSIGHT_API_TOKEN

// Before the heartbeat — inject context from prior sessions
const memories = await recall({
  companyId,
  agentId,
  query: `${task.title}\n${task.description}`,
}, config)

if (memories) {
  systemPrompt = `Past context:\n${memories}\n\n${systemPrompt}`
}

// After the heartbeat — store what the agent did
await retain({
  companyId,
  agentId,
  content: agentOutput,
  documentId: runId,
}, config)
```

Get an API key at [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup).

## How It Works

```
Paperclip Heartbeat
        │
        ▼
   recall()              ← Query Hindsight for prior context
        │
        ▼
  Agent executes          ← Prompt enriched with memories
        │
        ▼
   retain()              ← Store output for future heartbeats
```

Memory is isolated per company and agent by default (`paperclip::{companyId}::{agentId}`), matching Paperclip's multi-tenant model.

## HTTP Adapter Integration

For agents running as HTTP webhook servers, use the Express middleware:

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
  res.json({ output })  // output is auto-retained by middleware
})
```

The middleware reads `agentId`, `companyId`, `runId`, and `context.taskDescription` from Paperclip's HTTP adapter request body, then auto-retains the agent's `output` field after each response.

## Process Adapter Integration

For agents running as scripts via Paperclip's Process adapter:

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

// ... agent executes ...

await retain({
  agentId: PAPERCLIP_AGENT_ID!,
  companyId: PAPERCLIP_COMPANY_ID!,
  content: agentOutput,
  documentId: PAPERCLIP_RUN_ID!,
}, config)
```

## Bank ID Isolation

By default, each company+agent pair gets its own memory bank:

| Setting | Bank ID format |
|---|---|
| Default | `paperclip::{companyId}::{agentId}` |
| Company-only | `paperclip::{companyId}` |
| Agent-only | `paperclip::{agentId}` |
| Custom prefix | `{prefix}::{companyId}::{agentId}` |

```typescript
// Shared memory across all agents in a company
loadConfig({ bankGranularity: ['company'] })

// Agent's global memory across all companies
loadConfig({ bankGranularity: ['agent'] })

// Custom prefix
loadConfig({ bankIdPrefix: 'myapp' })
```

## Configuration

| Option | Env Variable | Default | Description |
|---|---|---|---|
| `hindsightApiUrl` | `HINDSIGHT_API_URL` | Required | Hindsight server URL |
| `hindsightApiToken` | `HINDSIGHT_API_TOKEN` | — | API token for Hindsight Cloud |
| `bankGranularity` | — | `['company', 'agent']` | Which IDs to include in the bank ID |
| `bankIdPrefix` | — | `'paperclip'` | Prefix for bank IDs |
| `recallBudget` | — | `'mid'` | Search depth: `low`, `mid`, or `high` |
| `recallMaxTokens` | — | `1024` | Max tokens in recalled memory block |
| `retainContext` | — | `'paperclip'` | Provenance label stored with memories |
| `timeoutMs` | — | `15000` | Request timeout in milliseconds |

## Skill File

A markdown skill file is included at `src/skills/hindsight.md`. Inject it into your agent's system prompt to give the agent direct access to Hindsight's REST API via `curl` for mid-task recall and retention.

## Requirements

- Node.js 20+ (uses native `fetch`, no external HTTP dependencies)
- Hindsight server (self-hosted or [Hindsight Cloud](https://hindsight.vectorize.io))
