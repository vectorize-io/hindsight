---
sidebar_position: 7
title: "Cline Persistent Memory with Hindsight | Integration Guide"
description: "Add persistent memory to the Cline VS Code coding agent with Hindsight using lifecycle hooks — no MCP. Hooks automatically recall context before each task and retain what happened after."
---

# Cline

[View Changelog →](/changelog/integrations/cline)

Persistent memory for [Cline](https://github.com/cline/cline) using [Hindsight](https://vectorize.io/hindsight) — **without MCP**. Cline's [lifecycle hooks](https://docs.cline.bot/customization/hooks) run small scripts that automatically recall relevant context before each task and retain what happened when a task ends. Because it runs on hooks, memory is deterministic — it doesn't depend on the model deciding to call a tool.

## Quick Start

:::tip Recommended: Hindsight Cloud
[Sign up free](https://ui.hindsight.vectorize.io/signup) for a Hindsight Cloud API key — no self-hosting required.
:::

Install the CLI, then run the installer from your project directory with your Hindsight URL and key:

```bash
pip install hindsight-cline

hindsight-cline install \
  --api-url https://api.hindsight.vectorize.io --api-token YOUR_KEY
```

Use `hindsight-cline install --global` to install for all projects, or `hindsight-cline uninstall` to remove it.

Then **enable hooks in Cline**: Settings → Features → Hooks.

:::note Platform
Cline hooks run on **macOS and Linux only** (no Windows) and require Python 3.
:::

## How It Works

| Cline hook         | What Hindsight does                                                   |
| ------------------ | --------------------------------------------------------------------- |
| `TaskStart`        | Recall context for the new task and inject it.                        |
| `UserPromptSubmit` | Recall memories for your message; record the prompt for later retain. |
| `TaskComplete`     | Retain the task's transcript and summary.                             |
| `TaskCancel`       | Retain the partial transcript of a cancelled task.                    |

Recalled memories are injected as a `<hindsight_memories>` context block. Cline doesn't hand hooks a transcript, so the integration accumulates each task's prompts locally and retains them at task end. Memories land in a single bank (`cline` by default).

## Configuration

Defaults live in the installed `settings.json`; put personal overrides in `~/.hindsight/cline.json` (stable across reinstalls), or use `HINDSIGHT_*` environment variables. Common keys: `hindsightApiUrl`, `hindsightApiToken`, `bankId`, `autoRecall`, `autoRetain`, `recallBudget`, `dynamicBankId`, `debug`.

## Self-Hosting

```bash
pip install hindsight-all
export HINDSIGHT_API_LLM_API_KEY=your-openai-key
hindsight-api  # http://localhost:8888
```

Point the installer at it: `--api-url http://localhost:8888`.

See the [integration source](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/cline) for full details.
