---
title: "Guide: Add n8n Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, n8n, automation, memory]
description: "Add n8n memory with Hindsight using the @vectorize-io/n8n-nodes-hindsight community node, so any workflow can retain, recall, and reflect on long-term memory alongside your other n8n integrations."
image: /img/guides/guide-n8n-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add n8n Memory with Hindsight](/img/guides/guide-n8n-memory-with-hindsight.svg)

If you want **n8n memory with Hindsight**, the cleanest setup is the `@vectorize-io/n8n-nodes-hindsight` community node. It adds a single **Hindsight** node with three operations — **Retain**, **Recall**, and **Reflect** — that you drop into any workflow next to your existing Slack, Sheets, OpenAI, and other nodes. That gives your automations long-term memory instead of starting every run stateless.

This is a good fit for n8n because n8n is the connective tissue of automation: triggers from Gmail, Slack, Sheets, Stripe, and Notion, with actions across hundreds of apps. Until now each workflow run has been stateless. With the Hindsight node you retain facts as they emerge in a workflow, recall relevant context before an LLM step, and reflect to get a synthesized answer over everything a bank has learned.

This guide walks through installing the community node, creating a Hindsight API credential, using each operation, and a quick verification flow so you can confirm memory is actually being used. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. In n8n, go to **Settings → Community Nodes → Install** and enter `@vectorize-io/n8n-nodes-hindsight`.
> 2. Restart n8n; the **Hindsight** node appears in the node panel.
> 3. Create a **Hindsight API** credential with your API URL (defaults to Hindsight Cloud) and `hsk_...` key.
> 4. Add a **Hindsight** node and pick an operation — Retain, Recall, or Reflect — with a Bank ID.
> 5. Verify that a later run recalls what an earlier run retained.

## Prerequisites

Before you start, make sure you have:

- A running n8n instance (Cloud or self-hosted)
- A reachable Hindsight backend, [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server
- A Hindsight API key (`hsk_...`) if you are using Cloud

## Step 1: Install the community node

In n8n, go to **Settings → Community Nodes → Install** and enter:

```
@vectorize-io/n8n-nodes-hindsight
```

Or install it directly in a self-hosted n8n:

```bash
cd ~/.n8n/custom
npm install @vectorize-io/n8n-nodes-hindsight
```

Restart n8n and the **Hindsight** node appears in the node panel.

## Step 2: Create a Hindsight API credential

In n8n, create a new **Hindsight API** credential:

- **API URL**: `https://api.hindsight.vectorize.io` (the default, Hindsight Cloud) or your self-hosted URL
- **API Key**: your `hsk_...` key. Leave it blank for an unauthenticated self-hosted instance.

The credential applies the `Bearer` authorization header automatically, and it can be tested against the Hindsight `/health` endpoint, which works for both Cloud and self-hosted.

If you do not have a key yet, [sign up for Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) (free tier) and grab one from the dashboard, or [self-host](https://hindsight.vectorize.io/developer/installation) instead.

## Step 3: Add a Hindsight node and pick an operation

Add a **Hindsight** node to your workflow, select the credential you created, and choose one of the three operations. Every operation takes a **Bank ID** — a bank is created on first use if it does not exist, so you can name it after a user, a customer, or a workflow.

### Retain

Store content in a bank. Hindsight extracts structured facts asynchronously after the call returns.

| Field | Description |
|---|---|
| Bank ID | The memory bank to store in (auto-created on first use) |
| Content | Free text to retain |
| Tags | Comma-separated tags applied to the stored memory (for example `user:alex,scope:profile`) |

### Recall

Search a bank for memories relevant to a query. Returns a `results` array.

| Field | Description |
|---|---|
| Bank ID | Memory bank to search |
| Query | Natural-language query |
| Budget | `low` / `mid` / `high` — controls how exhaustive the retrieval is (defaults to `mid`) |
| Max Tokens | Maximum tokens of memory to return (defaults to `4096`) |
| Tags Filter | Comma-separated tags to filter recall (leave blank for no filter) |

### Reflect

Get an LLM-synthesized answer over the bank's accumulated knowledge. Returns `text`.

| Field | Description |
|---|---|
| Bank ID | Memory bank to reflect on |
| Query | Question to answer using the bank's memories |
| Budget | `low` / `mid` / `high` (defaults to `mid`) |

## What you get

The node ships with zero runtime dependencies and uses n8n's built-in authenticated HTTP helper, so it works on both self-hosted n8n and n8n Cloud. Because the three operations are ordinary n8n nodes, you can wire them anywhere in a workflow:

- **Retain** every closed support ticket, sales-call summary, or form submission into a bank
- **Recall** relevant context before an OpenAI, Anthropic, or Cohere step so the LLM sees prior history
- **Reflect** to ask a synthesizing question ("What do we know about this customer?") right inside a workflow

For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Verify that memory is working

A good test sequence is:

1. add a **Hindsight** node set to **Retain**, with a Bank ID like `demo-bank` and some Content
2. run the workflow so the content is retained
3. add a second **Hindsight** node set to **Recall**, pointed at the same Bank ID
4. give it a Query that relates to what you retained
5. run it and check the `results` array

For example, retain a note about a customer's plan, then recall on the customer's name in a later run. If the recall surfaces the earlier note, the setup is working.

## Common mistakes

### Forgetting to restart n8n after install

The **Hindsight** node only appears in the node panel after you restart n8n following the community-node install.

### Mismatched Bank IDs

Retain and Recall must use the same Bank ID to see the same memory. A typo creates a new, empty bank on first use rather than erroring.

### Testing recall too early

Hindsight extracts facts asynchronously after a Retain call returns. If you recall immediately, the just-retained content may not be fully processed yet.

### Wrong API URL for self-hosted

The credential defaults to Hindsight Cloud. If you self-host, set the API URL to your deployment; leave the API Key blank for an unauthenticated self-hosted instance.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — set the credential's API URL to your deployment, and leave the API Key blank if it is unauthenticated.

### Where do I install the node?

In n8n under **Settings → Community Nodes → Install**, or via `npm install @vectorize-io/n8n-nodes-hindsight` in a self-hosted `~/.n8n/custom` directory.

### How is memory scoped?

By Bank ID. Each operation takes a Bank ID, and a bank is created on first use, so you can isolate memory per user, customer, or workflow.

### What can I build with it?

Anything that benefits from persistent context — a customer-support assistant that recalls similar past tickets, a sales-call coach that pulls prior touchpoints, or a Slack bot that remembers conversations across sessions.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [n8n integration docs](https://hindsight.vectorize.io/docs/integrations/n8n)
