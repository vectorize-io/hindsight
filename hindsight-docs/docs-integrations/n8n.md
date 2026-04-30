---
sidebar_position: 32
title: "n8n Persistent Memory with Hindsight | Integration"
description: "Add persistent long-term memory to any n8n workflow with Hindsight. A community node provides Retain, Recall, and Reflect operations — drop them into any workflow alongside Slack, Sheets, OpenAI, and 400+ other integrations."
---

# n8n

Persistent memory for [n8n](https://n8n.io) workflows via [Hindsight](https://hindsight.vectorize.io). The `@vectorize-io/n8n-nodes-hindsight` community node package adds three operations — **Retain**, **Recall**, **Reflect** — that work alongside any other n8n node.

## Why this matters

n8n is the connective tissue of automation: triggers from Gmail, Slack, Sheets, Stripe, Notion; actions across 400+ apps. Until now it has been **stateless** — every workflow run starts fresh. With Hindsight nodes you can:

- **Retain** every closed support ticket, sales-call summary, or form submission into a memory bank
- **Recall** relevant context before an OpenAI / Anthropic / Cohere step so the LLM sees prior history
- **Reflect** to ask synthesizing questions ("What do we know about this customer?") right inside a workflow

## Installation

In your n8n instance, go to **Settings → Community Nodes → Install** and enter:

```
@vectorize-io/n8n-nodes-hindsight
```

Or for self-hosted n8n:

```bash
cd ~/.n8n/custom
npm install @vectorize-io/n8n-nodes-hindsight
```

Restart n8n; the **Hindsight** node appears in the node panel.

## Setup

1. **Sign up** at [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) (free tier) or [self-host](/developer/installation)
2. **Get an API key** from the Hindsight dashboard
3. **In n8n**, create a new **Hindsight API** credential with your API URL (defaults to Hindsight Cloud) and the `hsk_...` key

## Operations

### Retain

Store content in a bank. Hindsight extracts facts asynchronously after the call returns.

| Field | Description |
|---|---|
| Bank ID | Memory bank to store in (auto-created on first use) |
| Content | Free text to retain |
| Tags | Comma-separated tags |

### Recall

Search a bank for memories relevant to a query. Returns a `results` array.

| Field | Description |
|---|---|
| Bank ID | Memory bank to search |
| Query | Natural-language query |
| Budget | `low` / `mid` / `high` |
| Max Tokens | Cap on returned memory tokens |
| Tags Filter | Filter by tag |

### Reflect

Get an LLM-synthesized answer over the bank. Returns `text`.

| Field | Description |
|---|---|
| Bank ID | Memory bank |
| Query | Question to answer |
| Budget | `low` / `mid` / `high` |

## Example workflows

**Customer-support assistant** — every closed Zendesk ticket retains the resolution. Every new ticket starts with a recall against the bank to surface similar past issues, then passes that context to OpenAI to draft the first reply.

**Sales-call coach** — Gong webhook → Hindsight Retain (call summary). Before each next prep call, recall on the prospect's name to pull every prior touchpoint, then format into the daily prep doc.

**Personal Slack bot** — Slack DM trigger → Hindsight Recall on the user's question → OpenAI for the answer → Slack reply. The bot remembers every conversation across sessions.

## Source

- npm: [`@vectorize-io/n8n-nodes-hindsight`](https://www.npmjs.com/package/@vectorize-io/n8n-nodes-hindsight)
- GitHub: [`hindsight-integrations/n8n`](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/n8n)
