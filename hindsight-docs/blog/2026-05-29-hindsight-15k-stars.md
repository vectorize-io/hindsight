---
title: "Hindsight Hits 15,000 Stars: What's Shipped Since 10k"
description: "Five weeks after the 10k milestone, Hindsight is at 15k GitHub stars. Here's what shipped in the open-source repo: 7 versions, 6 new integrations, and the Constellation graph view."
slug: "2026/05/29/hindsight-15k-stars"
date: 2026-05-29T12:00
image: "/img/blog/hindsight-15k-stars.png"
tags: [release, milestone]
hide_table_of_contents: true
---

![Hindsight Hits 15,000 Stars](/img/blog/hindsight-15k-stars.png)

Five weeks ago, [Hindsight crossed 10,000 GitHub stars](/blog/2026/04/22/hindsight-10k-stars). Today, the counter rolled past 15,000.

That's +5,000 stars and +247 forks in roughly five weeks. The acceleration is the part worth noting — it means new people are finding Hindsight every day, and a steady share of them are sticking around long enough to fork, contribute, or file thoughtful bugs.

We didn't want to write a second "thanks for the stars" post. So instead, here's the short version of what actually shipped in the open-source repo between 10k and 15k.

<!-- truncate -->

---

## Seven Releases in Five Weeks

[v0.5.3](/blog/2026/04/17/version-0-5-3), [v0.5.5](/blog/2026/04/28/version-0-5-5), [v0.6.0](/blog/2026/05/05/version-0-6-0), [v0.6.1](/blog/2026/05/08/version-0-6-1), [v0.6.2](/blog/2026/05/14/version-0-6-2), [v0.7.0](/blog/2026/05/27/version-0-7-0), and [v0.7.1](/blog/2026/05/28/version-0-7-1) all landed in this window. Highlights:

- **0.6.0** removed all process-environment reads from the integration plugins. Credentials now flow through `SecretRef` (env, file, or exec sources) — a meaningful security upgrade for production deployments.
- **0.7.0** shipped the largest API surface change since v0.5: refined consolidation, reflect improvements, and mental model refresh hooks.

## Six New Integrations

| Integration | Post |
|---|---|
| OpenAI Agents | [Persistent memory for OpenAI Agents SDK](/blog/2026/04/17/openai-agents-persistent-memory) |
| Pipecat | [Voice AI with long-term memory](/blog/2026/04/28/pipecat-voice-ai-persistent-memory) |
| smolagents | [Tool-based memory for Hugging Face's smolagents](/blog/2026/04/29/smolagents-memory-tools) |
| AWS AgentCore | [Multi-turn memory for Bedrock Agents](/blog/2026/05/01/agentcore-persistent-memory) |
| n8n | [The memory layer every n8n workflow was missing](/blog/2026/05/07/n8n-persistent-memory) |
| Paperclip | [Adding persistent memory to Paperclip agents](/blog/2026/05/26/paperclip-persistent-memory) |

That brings the official integration count to 27.

## The Constellation Graph View

The [Constellation view](/blog/2026/04/16/constellation-view) — an interactive graph of your memories and the entities connecting them — shipped just before the 10k milestone but earned most of its usage in this window. The same engine now powers the Entities → Relations tab. Ships with the open-source control plane, runs against any self-hosted instance.

## A Few Posts Worth Re-Reading

Some of the most-shared writing from the past five weeks:

- [Your Agent Isn't Forgetful — It Was Never Given a Memory](/blog/2026/04/23/your-agent-is-not-forgetful) — the foundational argument
- [Why Every Agent Harness Needs a Memory Layer](/blog/2026/05/04/agent-harness-needs-memory) — the broader case for memory as infrastructure
- [The Case Against External Vector DBs for Agent Memory](/blog/2026/05/12/case-against-external-vector-dbs-agent-memory) — why we don't ship one
- [The Consolidation Problem in Agent Memory](/blog/2026/05/21/agent-memory-consolidation) — keep, merge, decay, evict
- [Building a Hermes Coding Assistant That Remembers Your Codebase](/blog/2026/05/25/hermes-coding-assistant-codebase-memory) — the coding-agent template

## What's Next

The 0.7 line still has a few releases left. Self-hosted users should watch for the next round of operations dialog improvements and a few performance wins in the recall path that didn't make it into 0.7.1. On the integrations side, we're closing the gap on the remaining major frameworks — the goal is "if you can run an agent, you can plug in Hindsight" by the end of Q3.

Thanks for getting us here. If you starred along the way, you're part of why.

[Star Hindsight on GitHub](https://github.com/vectorize-io/hindsight) or [self-host with a single Docker command](https://hindsight.vectorize.io/developer/installation).
