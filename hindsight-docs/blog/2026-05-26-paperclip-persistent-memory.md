---
title: "Adding Persistent Memory to Paperclip Agents with Hindsight"
authors: [benfrank241]
date: 2026-05-26T12:00
tags: [integrations, paperclip, agents, memory, guide, tutorial]
description: "Give every Paperclip agent long-term memory with one plugin install. Agents recall context before runs and retain learnings after, no code changes needed."
image: /img/blog/paperclip-persistent-memory.png
hide_table_of_contents: true
---

![Adding Persistent Memory to Paperclip Agents with Hindsight](/img/blog/paperclip-persistent-memory.png)

[Paperclip](https://github.com/paperclipai/paperclip) treats AI agents like employees. You define an org chart, assign roles, set budgets, and let agents work autonomously within that structure. 67,000+ GitHub stars since March 2026. Five adapter types. A plugin system that extends every agent without touching their code.

One thing the org chart doesn't give agents: a memory that survives between runs.

Every Paperclip run starts from zero. The agent that triaged a customer issue yesterday has no idea it ran yesterday. The coding agent that discovered a flaky test last week will discover it again this week. The knowledge your agents accumulate during a run evaporates the moment the run ends.

The Hindsight plugin fixes this. Install it once, configure it in Settings, and every agent in your Paperclip instance gets persistent [agent memory](https://vectorize.io/what-is-agent-memory/) that compounds across runs, companies, and restarts. No code changes to any agent.

<!-- truncate -->

## Why Paperclip Agents Need Memory

Paperclip's company model makes the memory gap worse than it is in single-agent frameworks.

Consider a four-agent setup: a support agent, a docs agent, a triage agent, and a code-review agent. Each agent handles dozens of runs per week. Each run produces decisions, discoveries, and context that would make the next run faster and better. Without memory, all of that vanishes.

The support agent re-diagnoses the same issue it resolved three days ago. The code-review agent flags the same false positive it already learned to ignore. The triage agent routes a ticket to the wrong team because it doesn't remember the reorg that happened last Tuesday.

In a human company, employees remember. They build institutional knowledge. The Paperclip model gives agents roles, reporting lines, and budgets, but without a memory layer, those agents restart as interns every single run.

## How the Plugin Works

The Hindsight plugin hooks into Paperclip's event system. No wrapper code, no agent modifications. Three lifecycle events drive the memory flow:

**`agent.run.started`**—Before an agent begins work, the plugin fetches the run's issue via the Paperclip SDK, builds a recall query from the issue title and description, and injects the most relevant memories into the plugin state for that run. The agent starts with context from every prior run that touched similar issues.

**`issue.comment.created`**—Every comment on an issue (user-authored or agent-authored) is retained to Hindsight automatically. The plugin fetches the full comment body, attributes it to the agent or falls back to the issue assignee, and stores it in the appropriate memory bank.

**`agent.run.finished`**—Currently a no-op. The subscription is kept so future Paperclip payload additions can be picked up without a plugin update.

By default, memory is keyed to `companyId` + `agentId`, never to the run ID. That's why it survives: it's scoped to the agent's role in the company, not to a single execution. The `bankGranularity` setting lets you change this scoping — narrowing it to per-user isolation or broadening it to a shared bank across all agents.

### Agent Tools for Mid-Run Use

Beyond the automatic recall-and-retain lifecycle, agents also get two tools they can call explicitly during a run:

- **`hindsight_recall(query)`**—Search memory for context relevant to a specific question. During a run, the tool returns the memories cached at startup. If no startup recall occurred (e.g., the run had no associated issue), the tool falls back to a live Hindsight query.
- **`hindsight_retain(content)`**—Store a fact, decision, or outcome immediately. Useful when an agent discovers something mid-run that shouldn't wait for the comment-based retention cycle.

## Install and Configure

### Prerequisites

You need a running Hindsight instance. Two options:

**Hindsight Cloud (recommended)**—[Sign up free](https://ui.hindsight.vectorize.io/signup), get an API key, and skip self-hosting entirely. The free tier is enough to test this integration end-to-end.

**Self-hosted**—If you prefer to run everything locally:

```bash
pip install hindsight-all
export HINDSIGHT_API_LLM_API_KEY=your-openai-key
hindsight-api
```

### Install the Plugin

One command:

```bash
pnpm paperclipai plugin install @vectorize-io/hindsight-paperclip
```

Then configure in **Settings > Plugins > Hindsight Memory**.

### Configuration Reference

| Field | Default | Description |
|-------|---------|-------------|
| `hindsightApiUrl` | `https://api.hindsight.vectorize.io` | Hindsight server URL (use `http://localhost:8888` for self-hosted) |
| `hindsightApiKeyRef` | — | Paperclip secret name holding your Hindsight API key |
| `dynamicBankId` | `true` | When true, bank ID is derived from `bankGranularity`. Set false and provide `bankId` to use a static shared bank |
| `bankId` | — | Static bank ID used when `dynamicBankId` is false. All agents sharing this value read/write the same memory bank |
| `bankGranularity` | `["company", "agent"]` | How memory banks are scoped when `dynamicBankId` is true (see below) |
| `recallBudget` | `mid` | `low` = fastest, `mid` = balanced, `high` = most thorough |
| `autoRetain` | `true` | Automatically retain issue comments after every comment event |

The plugin validates the connection on save by pinging the Hindsight health endpoint. If the URL or key is wrong, you'll know immediately.

### Adapter Compatibility

The plugin works through Paperclip's event system, not through the adapter layer. That means it works with every adapter type:

| Adapter | Supported |
|---------|-----------|
| Claude | Yes |
| Codex | Yes |
| Cursor | Yes |
| HTTP | Yes |
| Process | Yes |

## Memory Isolation: Choosing the Right Bank Granularity

The `bankGranularity` setting controls how memory banks are scoped. This is the most important architectural decision in the setup.

### Company + Agent (Default)

Bank ID format: `paperclip::{companyId}::{agentId}`

Each agent in each company gets its own memory bank. The support agent's memories are separate from the code-review agent's memories. Company A's support agent and Company B's support agent each have their own bank.

**Use this when:** agents have distinct roles and shouldn't cross-contaminate knowledge. This is the right default for most setups.

### Company Only

Bank ID format: `paperclip::{companyId}`

All agents in a company share one memory bank. The support agent's retained memories are visible to the triage agent's recall queries, and vice versa.

**Use this when:** you have a small team of agents that work on the same domain and benefit from shared context. A three-agent squad handling the same product area, for example.

### Agent Only

Bank ID format: `paperclip::{agentId}`

The agent's memory follows it across companies. If you run the same code-review agent across five different company environments, it remembers patterns from all five.

**Use this when:** you have a utility agent that should accumulate cross-company expertise. A style-guide enforcer, a compliance checker, or a shared knowledge base curator.

### Company + Agent + User

Bank ID format: `paperclip::{companyId}::{agentId}::user::{userId}`

Each user gets their own memory bank per agent per company. The plugin extracts user identity from the issue's `originId` (e.g., `slack::alice@acme.com`) or `creatorEmail` field. If no user can be identified for a given event, the bank ID falls back to company+agent.

**Use this when:** you need per-user memory isolation — for GDPR compliance, multi-tenant support platforms, or any case where one user's context shouldn't bleed into another's recall results.

### Static Shared Bank

Instead of deriving bank IDs dynamically, you can set `dynamicBankId: false` and provide a fixed `bankId`. Every agent in every company reads and writes the same bank.

**Use this when:** you have a multi-agent cohort that needs collaborative memory — a squad of agents working the same product area where shared context is more valuable than isolation.

## What Gets Remembered

### Automatic (No Agent Action Required)

**On run start:** The plugin fetches the issue associated with the run, constructs a query from the issue's title and description, and recalls relevant memories from the agent's bank. Those memories are cached in plugin state and made available to the agent via the `hindsight_recall` tool.

**On every comment:** When a comment is created on an issue (whether by a user or by the agent itself), the plugin fetches the full comment body and retains it to the agent's memory bank. This is the primary durable signal. Over time, it builds a complete record of every interaction the agent has participated in.

### Explicit (Agent-Initiated)

**`hindsight_retain(content)`**—The agent stores a learning, decision, or outcome mid-run. Example: the agent discovers that a particular API endpoint is deprecated. It retains that fact immediately so it's available on the next run, even if the comment thread doesn't capture it clearly.

**`hindsight_recall(query)`**—The agent searches memory for a specific topic. The startup recall uses the issue title and description as the query, which is broad. Mid-run, the agent can issue targeted queries: "What was the resolution for the Stripe webhook timeout issue?" or "What conventions does this team use for error handling?"

## The Compounding Effect

The first run looks the same as it did before the plugin, minus a few hundred milliseconds of recall latency. By the tenth run, the agent is pulling in context from nine prior runs. By the fiftieth, the agent has seen enough issues that its startup recall routinely surfaces the exact prior resolution for the problem at hand.

This is the shift from stateless to compounding. The agent's value doesn't reset between runs. It accumulates.

One plugin install. Zero code changes. Memory that grows with every run your agents complete.

---

**Further reading:**

- [What Is Agent Memory?](https://vectorize.io/what-is-agent-memory/), foundational concepts behind how AI agents retain context
- [Best AI Agent Memory Systems in 2026](https://vectorize.io/articles/best-ai-agent-memory-systems/), comparison of all major agent memory frameworks
- [Why Every Agent Harness Needs a Memory Layer](/blog/2026/05/04/agent-harness-needs-memory), the broader argument for persistent memory in agent runtimes
- [Paperclip integration docs](https://hindsight.vectorize.io/integrations/paperclip), full configuration reference and bank ID format details
