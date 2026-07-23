
# Dify

Persistent memory for [Dify](https://dify.ai) workflows via [Hindsight](https://hindsight.vectorize.io). The Hindsight Dify plugin adds three tools — **Retain**, **Recall**, **Reflect** — that work alongside any other Dify node in workflows, chatflows, and agent apps.

## Why this matters

Dify is one of the most popular open-source LLM app platforms — visual workflow builder, prompt management, and a growing tool/plugin ecosystem. Until now, Dify workflows have been **stateless across runs**. With Hindsight tools you can:

- **Retain** every closed support ticket, sales-call transcript, or form submission into a memory bank
- **Recall** relevant context before an LLM step so the model sees prior history
- **Reflect** to ask synthesizing questions ("What do we know about this customer?") inside a workflow

## Installation

In your Dify dashboard go to **Plugins → Install Plugin**, then choose one of:

- **Marketplace** — search for `Hindsight` (once published)
- **GitHub** — install from `vectorize-io/hindsight` (path `hindsight-integrations/dify`)
- **Local** — upload the `.difypkg` archive

After install, the **Hindsight** plugin appears under **Tools** in the workflow editor.

## Setup

> **💡 Recommended: Hindsight Cloud**
>
[Sign up free](https://ui.hindsight.vectorize.io/signup) and grab an API key — no self-hosting required.
1. **Sign up** at [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) (free tier) or [self-host](../../developer/installation.md)
2. **Get an API key** from the Hindsight dashboard
3. **In Dify**, open the Hindsight plugin and add credentials:
   - **API URL** — defaults to `https://api.hindsight.vectorize.io` (Cloud); change for self-hosted
   - **API Key** — your `hsk_...` key (optional for self-hosted unauthenticated instances)

## Tools

### Retain

Store content in a bank. Hindsight extracts facts asynchronously after the call returns.

| Field | Description |
|---|---|
| Bank ID | Memory bank to store in (auto-created on first use) |
| Content | Free-text content to retain |
| Tags | Optional comma-separated tags |

### Recall

Search a bank for memories relevant to a query. Returns a `results` array.

| Field | Description |
|---|---|
| Bank ID | Memory bank to search |
| Query | Natural-language query |
| Budget | `low` / `mid` / `high` |
| Max Tokens | Cap on returned tokens |
| Tags | Optional tag filter (comma-separated) |

### Reflect

Get an LLM-synthesized answer over the bank. Returns `text`.

| Field | Description |
|---|---|
| Bank ID | Memory bank |
| Query | Question to answer |
| Budget | `low` / `mid` / `high` |

## Example workflows

**Customer-support assistant** — every closed Zendesk ticket retains the resolution. Every new ticket starts with a Recall against the bank to surface similar past issues, then passes that context to OpenAI/Anthropic to draft the first reply.

**Sales-call coach** — Gong webhook → Hindsight Retain (call summary). Before each next prep call, Recall on the prospect's name to pull every prior touchpoint, then format into the daily prep doc.

**Knowledge-base agent** — every uploaded document is chunked and retained. The Dify chatflow uses Recall instead of vector-DB-only retrieval, getting fact-extracted, deduplicated, time-aware results.

## Source

- GitHub: [`hindsight-integrations/dify`](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/dify)
