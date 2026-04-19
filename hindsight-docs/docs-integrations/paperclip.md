---
sidebar_position: 11
title: "Paperclip Persistent Memory with Hindsight | Integration Guide"
description: "Add long-term memory to all Paperclip agents with Hindsight. Install once as a plugin — every agent gets automatic recall before runs and retain after runs."
---

# Paperclip

Persistent memory for [Paperclip AI](https://github.com/paperclipai/paperclip) agents using [Hindsight](https://hindsight.vectorize.io).

Install the `@vectorize-io/hindsight-paperclip` plugin once. Every agent in your Paperclip instance automatically gets long-term memory that persists across runs, companies, and restarts — no code changes required.

## Installation

```bash
pnpm paperclipai plugin install @vectorize-io/hindsight-paperclip
```

Then configure in **Settings → Plugins → Hindsight Memory**.

## Prerequisites

Either:

```bash
# Self-hosted
pip install hindsight-all
export HINDSIGHT_API_LLM_API_KEY=your-openai-key
hindsight-api
```

Or [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) — no self-hosting required.

## How It Works

```
agent.run.started
  └─ recall(issueTitle + description)
       └─ cached in plugin state for this run

agent running…
  ├─ hindsight_recall(query) → returns cached context or live recall
  └─ hindsight_retain(content) → stores immediately

agent.run.finished
  └─ retain(output) → stored with runId as document_id
```

Memory is keyed to `companyId` + `agentId` — never to the run ID — so it accumulates across every run.

## Configuration

| Field | Default | Description |
|-------|---------|-------------|
| `hindsightApiUrl` | `http://localhost:8888` | Hindsight server URL |
| `hindsightApiKeyRef` | — | Paperclip secret name holding Hindsight Cloud API key |
| `bankGranularity` | `["company", "agent"]` | Memory isolation: per company+agent, per company, or per agent |
| `recallBudget` | `mid` | `low` = fastest, `mid` = balanced, `high` = most thorough |
| `autoRetain` | `true` | Automatically retain run output after every run |

## Bank ID Format

```
paperclip::{companyId}::{agentId}    ← default (company + agent granularity)
paperclip::{companyId}               ← company granularity (shared across agents)
paperclip::{agentId}                 ← agent granularity (agent memory across companies)
```

## Agent Tools

Agents can call these tools directly during a run:

**`hindsight_recall(query)`** — search memory for relevant context. Called automatically at run start; agents can also call it mid-run for targeted queries.

**`hindsight_retain(content)`** — store a fact or decision immediately, without waiting for run end.

## Adapter Compatibility

Works with all Paperclip adapter types via the event system:

| Adapter | Supported |
|---------|-----------|
| Claude | ✓ |
| Codex | ✓ |
| Cursor | ✓ |
| HTTP | ✓ |
| Process | ✓ |
