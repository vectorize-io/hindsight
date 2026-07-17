---
title: "Guide: Add Dify Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, dify, agents, memory]
description: "Add Dify memory with Hindsight using a plugin that adds Retain, Recall, and Reflect tools you can drop into any Dify workflow, chatflow, or agent app."
image: /img/guides/guide-dify-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add Dify Memory with Hindsight](/img/guides/guide-dify-memory-with-hindsight.svg)

If you want **Dify memory with Hindsight**, the cleanest setup is the Hindsight Dify plugin. It adds three tools — Retain, Recall, and Reflect — that drop into any Dify workflow alongside your other LLM, search, and tool nodes. That gives Dify long-term memory across runs instead of forcing every workflow execution to start stateless.

This is a good fit for Dify because Dify is a visual workflow builder with a growing tool and plugin ecosystem, but its workflows are stateless across runs by default. The plugin uses the same node model as everything else in Dify: you place a Retain node to store content, a Recall node to pull relevant context before an LLM step, or a Reflect node to ask a synthesizing question over a bank.

This guide walks through installing the plugin, adding your Hindsight credentials, wiring the three tools into a workflow, and a quick verification flow so you can confirm memory is actually being used. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. In Dify, go to **Plugins → Install Plugin** and install the **Hindsight** plugin.
> 2. Open the Hindsight plugin and add your **API URL** and **API Key** credentials.
> 3. Add a **Recall** node before an LLM step to pull relevant context.
> 4. Add a **Retain** node to store new content into a bank.
> 5. Verify that a later run recalls what an earlier one stored.

## Prerequisites

Before you start, make sure you have:

- A Dify instance where you can install plugins and build workflows
- A reachable Hindsight backend, [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server
- A Hindsight API key from the dashboard (optional for self-hosted unauthenticated instances)

## Step 1: Install the plugin

In your Dify dashboard go to **Plugins → Install Plugin**, then choose one of:

- **Marketplace** — search for `Hindsight` (once published)
- **GitHub** — install from `vectorize-io/hindsight` (path `hindsight-integrations/dify`)
- **Local** — upload the `.difypkg` archive

After install, the **Hindsight** plugin appears under **Tools** in the workflow editor.

## Step 2: Add your Hindsight credentials

Open the Hindsight plugin and add credentials:

- **API URL** — defaults to `https://api.hindsight.vectorize.io` (Cloud); change it for a self-hosted server
- **API Key** — your `hsk_...` key (optional for self-hosted unauthenticated instances)

If you do not have a key yet, [sign up for Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) on the free tier and grab one from the dashboard, or [self-host](https://hindsight.vectorize.io/developer/installation) a server.

## Step 3: Wire the tools into a workflow

The plugin gives you three tools you place like any other Dify node.

- **Retain** — store free-text content in a bank. Fields are **Bank ID** (the memory bank to store in, auto-created on first use), **Content** (the free-text to retain), and optional **Tags** (comma-separated). Hindsight extracts facts asynchronously after the call returns.
- **Recall** — search a bank for memories relevant to a query. Fields are **Bank ID**, **Query** (natural language), **Budget** (`low` / `mid` / `high`), **Max Tokens** (cap on returned tokens), and optional **Tags**. It returns a `results` array.
- **Reflect** — get an LLM-synthesized answer over the bank. Fields are **Bank ID**, **Query** (the question to answer), and **Budget** (`low` / `mid` / `high`). It returns a `text` field.

A typical shape is a **Recall** node before an LLM step to surface prior history, then a **Retain** node after the run to store the new content back into the same bank.

For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## What you get

Because the tools are ordinary Dify nodes, you can drop memory into chatflows, workflows, and agent apps without leaving the visual builder:

- **Customer-support assistant** — every closed ticket retains the resolution. Every new ticket starts with a Recall against the bank to surface similar past issues, then passes that context to your LLM node to draft the first reply.
- **Sales-call coach** — a call summary is retained after each call. Before the next prep call, a Recall on the prospect's name pulls every prior touchpoint into the daily prep doc.
- **Knowledge-base agent** — uploaded documents are retained, and the chatflow uses Recall instead of vector-DB-only retrieval to get fact-extracted, deduplicated, time-aware results.

## Verify that memory is working

A good test sequence is:

1. run a workflow that ends with a **Retain** node writing to a bank
2. store a distinctive fact — for example a decision or a customer detail
3. run a second workflow with a **Recall** node reading the same **Bank ID**
4. query for the earlier fact
5. confirm the earlier fact appears in the `results` array

If the Recall node surfaces what the earlier run stored, the setup is working.

## Common mistakes

### Mismatched Bank IDs

Recall and Retain only share memory when they use the same **Bank ID**. If a later Recall returns nothing, check that it points at the bank the earlier Retain wrote to.

### Testing Recall immediately after Retain

Retain extracts facts asynchronously after the call returns. If you Recall the instant a Retain finishes, the facts may not be searchable yet.

### Missing or wrong credentials

If the tools error, confirm the **API URL** points at your backend and the **API Key** is set (required for Hindsight Cloud).

### Treating Reflect like Recall

Recall returns a `results` array of memories; Reflect returns a synthesized `text` answer. Use Recall to feed an LLM step and Reflect when you want the synthesized answer directly.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — point the plugin's **API URL** at it, and the API key is optional for unauthenticated instances.

### Where does the plugin appear in Dify?

After install, the **Hindsight** plugin shows up under **Tools** in the workflow editor.

### How is memory scoped?

By **Bank ID**. Each bank is an isolated memory store, and you choose which bank each tool node reads from or writes to.

### Which tool should I use?

Use **Retain** to store content, **Recall** to pull relevant memories before an LLM step, and **Reflect** to get an LLM-synthesized answer over a bank.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [Dify integration docs](https://hindsight.vectorize.io/docs/integrations/dify)
