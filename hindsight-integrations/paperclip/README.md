# @vectorize-io/hindsight-paperclip

Persistent long-term memory for Paperclip agents via [Hindsight](https://github.com/vectorize-io/hindsight).

Install once. Every agent in your Paperclip instance gets memory that persists across runs, companies, and restarts.

## What It Does

- **Before each run** — fetches the run's issue and recalls relevant memories on its title + description, caches them for the agent
- **After each comment** — retains the full comment body to Hindsight (durable record of both user and agent output)
- **Agent tools** — `hindsight_recall` and `hindsight_retain` tools for agents to query and store memory mid-run

## Installation

```bash
pnpm paperclipai plugin install @vectorize-io/hindsight-paperclip
```

Then configure in **Settings → Plugins → Hindsight Memory**.

## Prerequisites

> ✨ **Recommended:** [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) — sign up free, get an API key, and skip the self-hosting setup entirely.

**Self-hosting alternative** — run Hindsight locally:

```bash
pip install hindsight-all
export HINDSIGHT_API_LLM_API_KEY=your-openai-key
hindsight-api
```

## Configuration

| Field                | Default                              | Description                                                                                                                                                            |
| -------------------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hindsightApiUrl`    | `https://api.hindsight.vectorize.io` | Hindsight server URL (Cloud default; use `http://localhost:8888` for self-hosted)                                                                                      |
| `hindsightApiKeyRef` | —                                    | Paperclip secret name holding Hindsight Cloud API key                                                                                                                  |
| `dynamicBankId`      | `true`                               | When `true`, bank ID is derived from `bankGranularity`. Set `false` and provide `bankId` to share one static memory bank across agents                                 |
| `bankId`             | —                                    | Static bank ID used when `dynamicBankId` is `false`. All agents sharing this value read/write the same memory bank                                                     |
| `bankGranularity`    | `["company", "agent"]`               | Memory isolation when `dynamicBankId` is `true`: per company+agent, per company, or per agent. Add `"user"` for per-user memory isolation (useful for GDPR compliance) |
| `recallBudget`       | `mid`                                | `low` = fastest, `mid` = balanced, `high` = most thorough                                                                                                              |
| `autoRetain`         | `true`                               | Automatically retain run output after every run                                                                                                                        |

## Bank ID Format

```
paperclip::{companyId}::{agentId}                  ← default (company + agent granularity)
paperclip::{companyId}                             ← company granularity (shared across agents)
paperclip::{agentId}                               ← agent granularity (agent memory across companies)
paperclip::{companyId}::{agentId}::user::{userId}  ← user granularity (per-user isolation, GDPR-friendly)
{bankId}                                           ← static shared bank (dynamicBankId = false)
```

## Agent Tools

Agents can call these tools directly during a run:

**`hindsight_recall(query)`** — search memory for relevant context. Called automatically at run start; agents can also call it mid-run for targeted queries.

**`hindsight_retain(content)`** — store a fact or decision immediately, without waiting for run end.

## How It Works

```
agent.run.started
  └─ fetch issue via ctx.issues.get
       └─ recall(issueTitle + description) → cached in plugin state for the run

agent running…
  ├─ hindsight_recall(query) → returns cached context or live recall
  └─ hindsight_retain(content) → stores immediately

issue.comment.created
  └─ retain(full comment body via ctx.issues.listComments)
       └─ bank attribution: agent comment author when present; otherwise issue assignee

agent.run.finished
  └─ no-op (subscription kept for future use when payload carries output)
```

The bundled plugin manifest declares the `issues.read` and `issue.comments.read` capabilities needed by the new SDK calls, so Paperclip may prompt for these on first install or upgrade.

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
