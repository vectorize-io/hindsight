---
title: "Guide: Add Zapier Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, zapier, automation, memory]
description: "Add Zapier memory with Hindsight using the Hindsight Zapier app, so your Zaps can retain, recall, and reflect on long-term memory across 7,000+ apps."
image: /img/guides/guide-zapier-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add Zapier Memory with Hindsight](/img/guides/guide-zapier-memory-with-hindsight.svg)

If you want **Zapier memory with Hindsight**, the setup is the Hindsight Zapier app. It adds three actions — Retain, Recall, and Reflect — plus instant triggers that start a Zap when a memory event fires. That gives your Zaps long-term memory instead of leaving every automation stateless, so context can flow between Hindsight and the thousands of apps Zapier connects.

This is a good fit for Zapier because Zapier already wires together Gmail, Slack, Sheets, HubSpot, Notion, forms, and thousands more. On their own those Zaps forget everything between runs. With Hindsight steps you can store what happens, search prior history before an AI step, get a grounded answer inside the Zap, and even kick off a new Zap the moment a memory operation completes.

This guide walks through connecting your Hindsight account in Zapier, using the Retain, Recall, and Reflect actions, wiring up the memory-event triggers, and a quick verification flow so you can confirm memory is actually being used. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. Sign up at [Hindsight Cloud](https://hindsight.vectorize.io) (or self-host) and get an API key (`hsk_...`).
> 2. In Zapier, add a Hindsight step and connect your account with the API key.
> 3. Use the **Retain** action to store content in a memory bank.
> 4. Use the **Recall** or **Reflect** action to search that bank before an AI step.
> 5. Verify that a later Zap surfaces what an earlier one stored.

## Prerequisites

Before you start, make sure you have:

- A Zapier account
- A reachable Hindsight backend, [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server
- A Hindsight API key (`hsk_...`) from the dashboard — optional only for a self-hosted instance running without auth

## Step 1: Get a Hindsight API key

Sign up at [Hindsight Cloud](https://hindsight.vectorize.io) (free tier) or [self-host](https://hindsight.vectorize.io/docs) a server. Then grab an API key (`hsk_...`) from the Hindsight dashboard.

If you self-host and run without authentication, you can skip the key — the connection can be made with a blank key.

## Step 2: Connect Hindsight in Zapier

In Zapier, add a Hindsight step and connect your account. The connection asks for two fields:

- **API Key** — your Hindsight key (starts with `hsk_`), sent as a bearer token. Required for Hindsight Cloud; leave it blank for a self-hosted instance running without auth.
- **API URL** — defaults to Hindsight Cloud (`https://api.hindsight.vectorize.io`). Point it at your own instance for self-hosted (for example `http://localhost:8888`).

Once connected, every Hindsight step exposes a **Bank** field as a dynamic dropdown. You can pick an existing bank or type a new bank id — banks are created on first use.

## Step 3: Retain content into a bank

The **Retain** action stores content in a memory bank. Hindsight extracts facts asynchronously after the call returns.

| Field     | Description                                                           |
| --------- | --------------------------------------------------------------------- |
| Bank      | Memory bank to store in (dynamic dropdown; auto-created on first use) |
| Content   | Free text to retain                                                   |
| Context   | Optional context for the content                                      |
| Tags      | Comma-separated tags                                                  |
| Timestamp | When the content occurred (defaults to now)                           |

A typical use is a trigger that fires on some completed event — a closed ticket, a form submission, a call summary — feeding its text into **Retain** so it becomes part of the bank.

## Step 4: Recall or Reflect before an AI step

To pull context back out, use one of two search actions.

**Recall** searches a bank for memories relevant to a query and returns the matches:

| Field             | Description            |
| ----------------- | ---------------------- |
| Bank              | Memory bank to search  |
| Query             | Natural-language query |
| Budget            | `low` / `mid` / `high` |
| Tags / Tags Match | Optional tag filter    |

**Reflect** returns a single LLM-synthesized answer grounded in the bank's memories:

| Field  | Description            |
| ------ | ---------------------- |
| Bank   | Memory bank            |
| Query  | Question to answer     |
| Budget | `low` / `mid` / `high` |

Place a **Recall** step before an AI step so the model sees prior history, or use **Reflect** when you want a ready-made grounded answer inside the Zap. For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Step 5: Trigger Zaps from memory events

Beyond actions, the Hindsight app provides instant triggers (REST Hooks) that fire when a memory event completes in a bank:

| Trigger                      | Fires when                                                    |
| ---------------------------- | ------------------------------------------------------------- |
| **Retain Completed**         | An asynchronous retain finishes processing                    |
| **Consolidation Completed**  | Memory consolidation synthesizes observations / mental models |
| **Memory Defense Triggered** | The memory-defense filter redacts or blocks incoming content  |

These let you start a Zap the moment memory work finishes — for example, format the new observations from a **Consolidation Completed** trigger and post them to Slack so the team sees what the agent learned.

## What you get

Together these pieces make memory a first-class part of your automations:

- **Retain** every closed ticket, form submission, or call summary into a memory bank
- **Recall** relevant context before an AI step so the model sees prior history
- **Reflect** to get a synthesized, memory-grounded answer right inside a Zap
- **Trigger** a Zap the moment a memory operation completes

Because the **Bank** field is shared across actions and triggers, everything that reads and writes the same bank draws from the same memory.

## Verify that memory is working

A good test sequence is:

1. build a Zap with a **Retain** step that stores a distinctive piece of content into a test bank
2. run the Zap so the content is retained
3. build a second Zap with a **Recall** (or **Reflect**) step pointed at the same bank
4. query for the earlier content
5. confirm the recalled result surfaces what the first Zap stored

For example, retain a note that a specific customer prefers email over phone, then in a later Zap recall "how does this customer prefer to be contacted?" If the recalled memory surfaces the earlier note, the setup is working.

## Common mistakes

### Using different banks for retain and recall

Retain and recall must point at the same **Bank** to see the same memories. If you type slightly different bank ids, the recall Zap will look empty.

### Testing recall too early

Retain extracts facts asynchronously after the call returns. If you check recall immediately, the content may not be fully processed yet — wait for retain to finish, or trigger off **Retain Completed**.

### Leaving the API URL on Cloud for a self-hosted instance

The API URL defaults to Hindsight Cloud. If you self-host, point the URL at your own instance, and leave the API Key blank only if that instance runs without auth.

### Expecting triggers on a fully air-gapped instance

Triggers rely on your Hindsight instance making an outbound POST to Zapier's webhook URL. Any instance with outbound internet works; a fully air-gapped one cannot deliver trigger events.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — point the connection's **API URL** at your instance, and leave the **API Key** blank if it runs without auth.

### Which apps can I connect memory to?

Any of the apps Zapier supports. Because the Hindsight steps are ordinary Zapier actions and triggers, they compose with the thousands of apps in the Zapier ecosystem.

### What is the difference between Recall and Reflect?

**Recall** returns the memories that match your query. **Reflect** returns a single LLM-synthesized answer grounded in the bank's memories. Use Recall to feed context into your own AI step; use Reflect when you want a ready-made grounded answer.

### How is memory scoped?

Per memory bank. The **Bank** field on every action and trigger selects which bank you read from or write to, and banks are created on first use.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [Zapier integration docs](https://hindsight.vectorize.io/docs/integrations/zapier)
