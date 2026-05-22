# Hindsight for Dify

Persistent long-term memory for [Dify](https://dify.ai) workflows via [Hindsight](https://hindsight.vectorize.io).

Three tools — **Retain**, **Recall**, **Reflect** — drop into any Dify workflow alongside your other nodes.

- **Author**: Vectorize, Inc.
- **Repository**: <https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/dify>
- **Contact**: support@vectorize.io · [GitHub Issues](https://github.com/vectorize-io/hindsight/issues)

## Setup

> ✨ **Recommended: [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup)** — free tier, no self-hosting required. Sign up and grab an API key in under a minute.

1. **Sign up** at [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) (free tier) or [self-host](https://hindsight.vectorize.io/developer/installation).
2. **Get an API key** from the Hindsight dashboard.
3. **In Dify**, install this plugin (Marketplace or upload `.difypkg`), then add credentials:
   - **API URL**: `https://api.hindsight.vectorize.io` (Cloud) or your self-hosted URL
   - **API Key**: your `hsk_...` key (optional for self-hosted unauthenticated instances)

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
| Max Tokens | Cap on tokens returned |
| Tags | Optional comma-separated tag filter |

### Reflect

Get an LLM-synthesized answer over the bank. Returns a `text` field.

| Field | Description |
|---|---|
| Bank ID | Memory bank |
| Query | Question to answer |
| Budget | `low` / `mid` / `high` |

## Example workflow

**Customer-support assistant** — every closed Zendesk ticket triggers a Retain. Every new ticket starts with a Recall against the bank, then passes the context to your LLM node to draft a first reply.

## Source

- GitHub: [`vectorize-io/hindsight`](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/dify)
- Docs: [hindsight.vectorize.io/sdks/integrations/dify](https://hindsight.vectorize.io/sdks/integrations/dify)
