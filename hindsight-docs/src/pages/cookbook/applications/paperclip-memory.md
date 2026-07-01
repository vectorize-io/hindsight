---
sidebar_position: 17
---

# Paperclip Memory


:::info Complete Application
This is a complete, runnable application demonstrating Hindsight integration.
[**View source on GitHub →**](https://github.com/vectorize-io/hindsight-cookbook/tree/main/applications/paperclip-memory)
:::


Give every [Paperclip](https://paperclipai.com) agent in your instance persistent long-term memory via Hindsight. Install one plugin and every agent — Claude, Codex, Cursor, HTTP, Process — recalls relevant context before each run and retains its output afterwards. Zero changes to your agent code.

## What This Demonstrates

- **One plugin, every adapter** — works with Paperclip's Claude, Codex, Cursor, HTTP, and Process adapters
- **Recall before run** — `agent.run.started` event hooks `hindsight_recall` and caches results for the run
- **Retain after run** — `agent.run.finished` event automatically retains the agent's output
- **Agent-callable tools** — `hindsight_recall` and `hindsight_retain` for queries and writes mid-run
- **Per-company memory isolation** — bank ID format `paperclip::{companyId}::{agentId}` keeps tenants separated

## Architecture

```
agent.run.started
  └─ recall(issueTitle + description)
       └─ stored in plugin state for this run

agent running…
  ├─ hindsight_recall(query) → returns cached or live recall
  └─ hindsight_retain(content) → stores immediately

agent.run.finished
  └─ retain(output) → stored with runId as document_id
```

Memory survives session, run, and restart churn — it's keyed to `companyId` + `agentId`, not the run ID.

## Prerequisites

1. **Paperclip** running locally (`http://127.0.0.1:3100` by default)

2. **Hindsight** — pick one:

   ```bash
   # Self-hosted (free, runs locally)
   pip install hindsight-all
   export HINDSIGHT_API_LLM_API_KEY=your-openai-key
   hindsight-api
   ```

   Or sign up for [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup).

3. **Node 20+** (only needed for the optional seed script in this folder)

## Quick Start

### 1. Install the Plugin

```bash
pnpm paperclipai plugin install @vectorize-io/hindsight-paperclip
```

### 2. Configure

In Paperclip: **Settings → Plugins → Hindsight Memory**

| Field | Value |
|---|---|
| `hindsightApiUrl` | `http://localhost:8888` (self-hosted) or your Cloud endpoint |
| `hindsightApiKeyRef` | Paperclip secret name holding your Hindsight Cloud API key (Cloud only) |
| `bankGranularity` | `["company", "agent"]` (default) |
| `recallBudget` | `mid` (`low` = fastest, `high` = most thorough) |
| `autoRetain` | `true` |

### 3. Run Any Agent

Once the plugin is enabled, any agent in any adapter automatically gets memory. Trigger a run from Paperclip's UI or API as you normally would. The first run is a cold start (no recall content); the second and onward will see prior context injected.

### 4. (Optional) Seed Memory and Inspect

To explore the memory cycle without waiting for several runs to accumulate, this folder includes a small Node script:

```bash
cd applications/paperclip-memory
npm install
HINDSIGHT_URL=http://localhost:8888 \
  COMPANY_ID=acme-corp \
  AGENT_ID=eng-agent \
  node seed-memory.mjs
```

It writes a few representative memories into `paperclip::acme-corp::eng-agent` so the next Paperclip run for that company+agent has something to recall.

## Calling the Tools From an Agent

Any Paperclip agent can call the plugin's tools directly during a run:

```ts
// Inside your Paperclip agent (Claude/Codex/Cursor/HTTP/Process)
const past = await tools.hindsight_recall({
  query: 'authentication PR review history',
});

// ...do work...

await tools.hindsight_retain({
  content: 'Decided to require MFA on /admin in PR #142.',
});
```

`hindsight_recall` is also called automatically at run start — agents only need to invoke it directly for follow-up queries with a different angle.

## Bank Granularity

| `bankGranularity` | Bank ID | Use When |
|---|---|---|
| `["company", "agent"]` (default) | `paperclip::{companyId}::{agentId}` | Each agent in each company has its own memory |
| `["company"]` | `paperclip::{companyId}` | All agents within a company share memory |
| `["agent"]` | `paperclip::{agentId}` | An agent's memory follows it across companies |

## Verifying Memory

After a few runs, query the bank directly to confirm content is being stored:

```bash
curl -s "http://localhost:8888/v1/default/banks/paperclip::acme-corp::eng-agent/memories/recall" \
  -H "Content-Type: application/json" \
  -d '{"query": "what has this agent worked on?"}'
```

## Common Issues

**"Plugin not found"**
- Confirm Paperclip is running and that `pnpm paperclipai plugin install @vectorize-io/hindsight-paperclip` completed.

**Recall returns nothing**
- First run for a given `companyId`+`agentId` always recalls empty. Run the agent at least twice, or use `seed-memory.mjs`.
- Check `bankGranularity` matches what you're querying — switching it changes the bank ID.

**"Connection refused"**
- Verify `hindsightApiUrl` in Paperclip's plugin config points at a running Hindsight instance.

---

**Built with:**
- [Paperclip](https://paperclipai.com) — multi-adapter agent runtime
- [@vectorize-io/hindsight-paperclip](https://www.npmjs.com/package/@vectorize-io/hindsight-paperclip) — Hindsight memory plugin for Paperclip
- [Hindsight](https://github.com/vectorize-io/hindsight) — Long-term memory for AI agents
