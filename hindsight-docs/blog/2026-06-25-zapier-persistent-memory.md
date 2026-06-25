---
title: "Zapier Persistent Memory: Actions and Triggers"
authors: [benfrank241]
slug: "2026/06/25/zapier-persistent-memory"
date: 2026-06-25T13:00
tags: [hindsight, zapier, integration, memory, agents, workflows, no-code, tutorial]
description: "The Hindsight app for Zapier adds persistent memory to any Zap: Retain, Recall, and Reflect actions, plus triggers that start Zaps from memory events."
image: /img/blog/zapier-persistent-memory.png
hide_table_of_contents: true
---

![Zapier Persistent Memory with Hindsight](/img/blog/zapier-persistent-memory.png)

[Zapier](https://zapier.com) connects the thousands of apps most teams already run on (8,000+ by Zapier's count): Slack, Gmail, HubSpot, Notion, Sheets, Airtable, OpenAI. You wire them together into Zaps, and a trigger in one app sets off actions in others. What no Zap carries is memory. Every run starts from zero, so the workflow that triaged a support ticket this morning has no idea it ran yesterday, and the LLM step in the middle of your Zap only knows what you stuffed into that one prompt.

The Hindsight app for Zapier closes that gap, and it does it in both directions. Hindsight shows up as **three actions** your Zaps can call (Retain, Recall, Reflect) and **three triggers** that start a Zap when something happens in your [agent memory](https://vectorize.io/what-is-agent-memory). Memory becomes both a tool your automations use and an event source that sets them off.

<!-- truncate -->

## TL;DR

- The Hindsight Zapier app gives any Zap access to persistent memory, across [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) or a self-hosted instance.
- **Three actions:** Retain (store content), Recall (search a bank), Reflect (an LLM-synthesized, memory-grounded answer).
- **Three triggers** (instant, via REST Hooks): Retain Completed, Consolidation Completed, Memory Defense Triggered. These let a memory event *start* a Zap. They use Hindsight's webhook API, which is available on self-hosted instances and on Hindsight Cloud's enterprise plan; it isn't enabled on Cloud by default. The three actions work on any plan.
- The Bank field is a dynamic dropdown; type a new bank id and it's created on first use.
- API-key auth (`hsk_...` as a Bearer token); leave the key blank to point at an open self-hosted instance.

---

## Why Zapier Zaps Need Persistent Memory

A Zap is stateless by design. A trigger fires, data flows through a few action steps, the run ends, and nothing about it survives to the next run. That's exactly what you want for "new Stripe charge → add a row to Sheets." It's a real limitation for the workflows people increasingly build in Zapier:

- **Customer-facing Zaps** that should treat the same person consistently across runs, not reintroduce themselves every time.
- **AI steps** (the OpenAI or Anthropic action in the middle of a Zap) that would be far better with relevant prior context than with a single hand-written prompt.
- **Operational pipelines** where every run could be informed by every run before it.

Without memory, the LLM step in your Zap knows only what you cram into that one prompt. With memory, it gets a recall of what's relevant to the current input, built up across every prior run. That's the difference between a Zap that repeats and a workflow that compounds.

## What the Hindsight Zapier App Adds

The app installs as a single **Hindsight** app in Zapier, with actions and triggers you drop into Zaps like any other.

### Actions

**Retain** stores content into a memory bank (`POST /memories`). Put a Retain step anywhere a Zap produces something worth remembering: a closed ticket summary, a call transcript, a form submission. You hand it free text; Hindsight's retain pipeline asynchronously extracts structured facts, deduplicates against the bank, and updates the bank's mental models, without blocking the Zap.

**Recall** searches a bank with a natural-language query (`POST /memories/recall`). Put a Recall step before any action that benefits from context, most often right before an LLM step, and feed it the trigger data or the user's current message. It returns the most relevant memories.

**Reflect** goes one step further (`POST /reflect`): instead of raw memories, it returns an LLM-synthesized, memory-grounded answer. Use it when you want a direct answer ("what do we know about this account?") rather than a list of snippets to pass downstream.

| Action | Endpoint | Use it for |
|---|---|---|
| Retain | `POST /memories` | Store content worth remembering |
| Recall | `POST /memories/recall` | Fetch relevant memories before an LLM step |
| Reflect | `POST /reflect` | Get a synthesized, grounded answer |

### Triggers

This is what sets the Zapier app apart from a call-it-as-a-tool integration: Hindsight can *start* a Zap. Each trigger subscribes to Hindsight's webhook API (`POST /webhooks`) using Zapier's instant REST Hooks, and is torn down cleanly when you turn the Zap off (`DELETE /webhooks/{id}`).

One availability note up front: the webhook API the triggers depend on is an **enterprise feature on Hindsight Cloud** and isn't enabled by default; it's available out of the box on self-hosted instances. The three actions above need no special plan.

| Trigger | Event | Fires when |
|---|---|---|
| Retain Completed | `retain.completed` | A retain finishes processing and facts are live in the bank |
| Consolidation Completed | `consolidation.completed` | Hindsight consolidates a bank's memories |
| Memory Defense Triggered | `memory_defense.triggered` | A memory-defense check flags something |

So you can build Zaps the other way around: when a retain completes, post the new facts to Slack; when consolidation finishes, kick off a downstream report; when memory defense triggers, open a ticket. Memory stops being only a destination and becomes a source of events.

Every action and trigger has a **Bank** field rendered as a dynamic dropdown (populated from your existing banks). You can also type a new bank id; banks are created on first use.

## Authentication

The app uses API-key auth:

- **API Key**: your Hindsight key (starts with `hsk_`), sent as `Authorization: Bearer <key>`. Required for Hindsight Cloud. Leave it blank for a self-hosted instance running without auth.
- **API URL**: defaults to Hindsight Cloud (`https://api.hindsight.vectorize.io`); point it at your own instance (for example `http://localhost:8888`) for self-hosting.

Triggers work self-hosted, where the webhook API is available out of the box. (On Hindsight Cloud, webhooks are an enterprise feature and aren't enabled by default.) They rely on your instance making an *outbound* POST to Zapier's webhook URL, which works for any box with outbound internet; only fully air-gapped instances can't. Each trigger registers its webhook with a freshly generated HMAC secret and verifies the `X-Hindsight-Signature: sha256=<hmac>` header on every delivery, rejecting any payload whose signature doesn't match.

## A Few Zaps to Build

- **Support that learns.** Trigger on a closed Zendesk or Intercom ticket → Retain the resolution. On the next ticket from that customer, Recall before the LLM drafts a reply.
- **Sales context on tap.** After a Gong or Fireflies call, Retain the transcript. Before the next meeting, Reflect to get a synthesized brief on the account.
- **Memory-driven notifications** (trigger-based, so it needs the webhook API: self-hosted or Cloud enterprise). Use the Retain Completed trigger to post newly learned facts into a Slack channel, so the team sees what the system is learning.
- **Grounded form responses.** A Typeform submission triggers a Recall, an LLM step drafts a personalized reply, and an email action sends it.

## Setup

1. Sign up at [hindsight.vectorize.io](https://ui.hindsight.vectorize.io/signup); the free tier is enough to start.
2. Grab an API key (`hsk_...`) from the dashboard.
3. In Zapier, add a Hindsight action or trigger to a Zap and connect your account with the API key (and API URL, if self-hosting).
4. Pick a bank from the dropdown (or type a new id), and you're storing and recalling memory inside your Zaps.

## Frequently Asked Questions

**Does Zapier have built-in memory across runs?**
No. Each Zap run is stateless. Persistent memory across runs comes from a memory layer like Hindsight wired into the Zap.

**Can a memory event start a Zap?**
Yes, via the app's triggers (Retain Completed, Consolidation Completed, Memory Defense Triggered), which fire instantly via REST Hooks. They depend on Hindsight's webhook API, which is available self-hosted and on Hindsight Cloud's enterprise plan; it isn't enabled on Cloud by default. The three actions work on any plan.

**Does it work with self-hosted Hindsight?**
Yes. Set the API URL to your instance; if it runs without auth, leave the API key blank. Triggers work as long as your instance can make outbound requests to Zapier.

**What's the difference between Recall and Reflect?**
Recall returns the most relevant memories for a query; Reflect returns an LLM-synthesized answer grounded in those memories. Use Recall to feed a downstream step, Reflect when you want the answer directly.

## Further reading

- [What is agent memory?](https://vectorize.io/what-is-agent-memory): the foundational concepts behind recall, retention, and memory banks.
- [Best AI agent memory systems](https://vectorize.io/articles/best-ai-agent-memory-systems): how the major agent memory frameworks compare.
- [The memory layer every n8n workflow was missing](/blog/2026/05/07/n8n-persistent-memory): the same idea for n8n's workflow engine.
