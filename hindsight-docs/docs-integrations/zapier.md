---
sidebar_position: 36
title: "Zapier Persistent Memory with Hindsight | Integration"
description: "Add long-term memory to your Zaps with Hindsight. Retain, Recall, and Reflect actions plus instant memory-event triggers connect Hindsight to 7,000+ apps."
---

# Zapier

Long-term agent memory for [Zapier](https://zapier.com) via [Hindsight](https://hindsight.vectorize.io). The Hindsight Zapier app adds three actions — **Retain**, **Recall**, **Reflect** — plus instant **triggers** that start a Zap when a memory event fires, so memory flows between Hindsight and 7,000+ apps.

## Why this matters

Zapier connects everything: Gmail, Slack, Sheets, HubSpot, Notion, forms, and thousands more. On its own, those Zaps are **stateless**. With Hindsight you can:

- **Retain** every closed ticket, form submission, or call summary into a memory bank
- **Recall** relevant context before an AI step so the model sees prior history
- **Reflect** to get a synthesized, memory-grounded answer right inside a Zap
- **Trigger** a Zap the moment a memory operation completes (e.g. notify Slack when consolidation finishes)

## Setup

:::tip Recommended: Hindsight Cloud
[Sign up free](https://ui.hindsight.vectorize.io/signup) and grab an API key — no self-hosting required.
:::

1. **Sign up** at [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) (free tier) or [self-host](/developer/installation)
2. **Get an API key** (`hsk_...`) from the Hindsight dashboard
3. **In Zapier**, add a Hindsight step and connect your account with the API key (the API URL defaults to Hindsight Cloud; point it at your own instance for self-hosted — leave the key blank if it runs without auth)

## Actions

### Retain

Store content in a bank. Hindsight extracts facts asynchronously after the call returns.

| Field     | Description                                                           |
| --------- | --------------------------------------------------------------------- |
| Bank      | Memory bank to store in (dynamic dropdown; auto-created on first use) |
| Content   | Free text to retain                                                   |
| Context   | Optional context for the content                                      |
| Tags      | Comma-separated tags                                                  |
| Timestamp | When the content occurred (defaults to now)                           |

### Recall (search)

Search a bank for memories relevant to a query.

| Field             | Description            |
| ----------------- | ---------------------- |
| Bank              | Memory bank to search  |
| Query             | Natural-language query |
| Budget            | `low` / `mid` / `high` |
| Tags / Tags Match | Optional tag filter    |

### Reflect (search)

Get an LLM-synthesized answer grounded in the bank's memories.

| Field  | Description            |
| ------ | ---------------------- |
| Bank   | Memory bank            |
| Query  | Question to answer     |
| Budget | `low` / `mid` / `high` |

## Triggers

Instant triggers (REST Hooks) that fire when a memory event completes in a bank:

| Trigger                      | Fires when                                                    |
| ---------------------------- | ------------------------------------------------------------- |
| **Retain Completed**         | An asynchronous retain finishes processing                    |
| **Consolidation Completed**  | Memory consolidation synthesizes observations / mental models |
| **Memory Defense Triggered** | The memory-defense filter redacts or blocks incoming content  |

## Example Zaps

**Support assistant** — a closed-ticket trigger (Zendesk) → Hindsight **Retain** the resolution. New ticket → Hindsight **Recall** similar past issues → OpenAI drafts the first reply.

**Memory digest** — Hindsight **Consolidation Completed** trigger → format the new observations → post to Slack so the team sees what the agent learned.

**Daily prep** — a calendar trigger → Hindsight **Reflect** ("What do we know about this prospect?") → append the answer to the prep doc.

## Source

- GitHub: [`hindsight-integrations/zapier`](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/zapier)
