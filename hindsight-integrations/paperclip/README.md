# @vectorize-io/hindsight-paperclip

Persistent long-term memory for Paperclip agents via [Hindsight](https://github.com/vectorize-io/hindsight).

Install once. Every agent in your Paperclip instance gets memory that persists across runs, companies, and restarts.

## What It Does

- **Before each run** — recalls relevant memories from past runs and caches them for the agent
- **After each run** — retains the agent's output to Hindsight automatically
- **Agent tools** — `hindsight_recall` and `hindsight_retain` tools for agents to query and store memory mid-run

## Installation

```bash
pnpm paperclipai plugin install @vectorize-io/hindsight-paperclip
```

Then configure in **Settings → Plugins → Hindsight Memory**.

## Prerequisites

Either:

```bash
# Self-hosted (runs locally)
pip install hindsight-all
export HINDSIGHT_API_LLM_API_KEY=your-openai-key
hindsight-api
```

Or [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) — no self-hosting required.

## Configuration

| Field                | Default                 | Description                                                    |
| -------------------- | ----------------------- | -------------------------------------------------------------- |
| `hindsightApiUrl`    | `http://localhost:8888` | Hindsight server URL                                           |
| `hindsightApiKeyRef` | —                       | Paperclip secret name holding Hindsight Cloud API key          |
| `bankGranularity`    | `["company", "agent"]`  | Memory isolation: per company+agent, per company, or per agent |
| `recallBudget`       | `mid`                   | `low` = fastest, `mid` = balanced, `high` = most thorough      |
| `autoRetain`         | `true`                  | Automatically retain run output after every run                |

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

## How It Works

```
agent.run.started
  └─ recall(issueTitle + description)
       └─ store in plugin state for this run (instant lookup by tools)

agent running…
  ├─ hindsight_recall(query) → returns cached context or live recall
  └─ hindsight_retain(content) → stores immediately

agent.run.finished
  └─ retain(output) → stored in Hindsight with runId as document_id
```

Memory is keyed to `companyId` + `agentId`, never to the Paperclip session or run ID — so it survives across any number of runs.

## Development

```bash
npm install
npm run build
npm test
```

Local install into a running Paperclip instance:

```bash
curl -X POST http://127.0.0.1:3100/api/plugins/install \
  -H "Content-Type: application/json" \
  -d '{"packageName":"/absolute/path/to/hindsight-integrations/paperclip","isLocalPath":true}'
```
