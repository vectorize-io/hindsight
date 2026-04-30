# @vectorize-io/n8n-nodes-hindsight

[Hindsight](https://hindsight.vectorize.io) memory nodes for [n8n](https://n8n.io) — give any n8n workflow persistent long-term memory with **retain**, **recall**, and **reflect** operations.

Drop a Hindsight node anywhere in a workflow to:

- **Retain** facts emerging from a workflow (form submissions, CRM updates, customer chat) into a memory bank
- **Recall** relevant context before an LLM step so the AI sees prior history
- **Reflect** to get an LLM-synthesized answer over the bank's accumulated knowledge

## Installation

In n8n, go to **Settings → Community Nodes → Install** and enter:

```
@vectorize-io/n8n-nodes-hindsight
```

Or install via npm in your self-hosted n8n:

```bash
cd ~/.n8n/custom
npm install @vectorize-io/n8n-nodes-hindsight
```

Restart n8n and the **Hindsight** node appears in the node panel.

## Setup

1. **Create a Hindsight account** at [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) (free tier available) — or self-host with the [Hindsight installer](https://hindsight.vectorize.io/developer/installation)
2. **Get an API key** from the Hindsight dashboard
3. **In n8n**, create a new **Hindsight API** credential:
   - **API URL**: `https://api.hindsight.vectorize.io` (or your self-hosted URL)
   - **API Key**: your `hsk_...` key (leave blank for unauthenticated self-hosted)

## Operations

### Retain — store content in a bank

| Field | Description |
|---|---|
| Bank ID | The memory bank to store in (auto-created on first use) |
| Content | Free text to retain. Hindsight extracts structured facts asynchronously |
| Tags | Comma-separated tags applied to the stored memory |

### Recall — search a bank

| Field | Description |
|---|---|
| Bank ID | Memory bank to search |
| Query | Natural-language query |
| Budget | `low` / `mid` / `high` — controls retrieval depth |
| Max Tokens | Cap on returned memory tokens |
| Tags Filter | Filter memories by tag |

Returns: `{ results: [{ text, score, ... }, ...] }`

### Reflect — LLM-synthesized answer

| Field | Description |
|---|---|
| Bank ID | Memory bank to reflect on |
| Query | Question to answer |
| Budget | `low` / `mid` / `high` |

Returns: `{ text: "...", citations: [...] }`

## Example workflows

**Customer-support assistant** — every closed Zendesk ticket retains the resolution; every new ticket starts with a recall against the bank to surface similar past issues.

**Sales-call coach** — Gong webhook → Hindsight Retain (call summary). Before each next prep call, recall on the prospect's name to pull every prior touchpoint.

**Personal Slack bot** — Slack DM trigger → Hindsight Recall on the user's question, pass through to OpenAI node, reply.

## License

MIT
