---
title: "Guide: Add Cline Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, cline, coding-agents, memory]
description: "Add Cline memory with Hindsight using lifecycle hooks that recall context before each task and retain what happened after — no MCP required."
image: /img/guides/guide-cline-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add Cline Memory with Hindsight](/img/guides/guide-cline-memory-with-hindsight.svg)

If you want **Cline memory with Hindsight**, the cleanest setup is the `hindsight-cline` installer, which wires Hindsight into Cline's lifecycle hooks. The hooks recall relevant memory before each task and retain what happened when a task ends. That gives Cline long-term memory across sessions instead of forcing every new task to rediscover the same project context.

This is a good fit for Cline because it does not rely on MCP. Cline exposes lifecycle hooks — small scripts that run at key moments — and the integration installs hooks on `TaskStart`, `UserPromptSubmit`, `TaskComplete`, and `TaskCancel`. Because it runs on hooks, memory is deterministic: it happens automatically and doesn't depend on the model deciding to call a tool.

This guide walks through installing the CLI, running the installer against your Hindsight backend, enabling hooks in Cline, and a quick verification flow so you can confirm memory is actually being used. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. `pip install hindsight-cline`.
> 2. Run `hindsight-cline install --api-url … --api-token …` from your project directory.
> 3. Enable hooks in Cline: Settings → Features → Hooks.
> 4. Recalled memory is injected as a `<hindsight_memories>` context block; memory lands in the `cline` bank by default.
> 5. Verify that a later task remembers what an earlier one stored.

## Prerequisites

Before you start, make sure you have:

- Cline installed and working in VS Code
- A reachable Hindsight backend, [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server
- macOS or Linux (Cline hooks do not run on Windows) with Python 3 available

## Step 1: Install the CLI

Install the `hindsight-cline` package.

```bash
pip install hindsight-cline
```

This gives you the `hindsight-cline` command, which installs and manages the Cline hook scripts for you.

## Step 2: Run the installer against Hindsight

From your project directory, run the installer with your Hindsight URL and key.

For Hindsight Cloud:

```bash
hindsight-cline install \
  --api-url https://api.hindsight.vectorize.io --api-token YOUR_KEY
```

Use `hindsight-cline install --global` to install for all projects, or `hindsight-cline uninstall` to remove it later (add `--global` if you installed globally).

The installer copies four hook scripts (`TaskStart`, `UserPromptSubmit`, `TaskComplete`, `TaskCancel`) plus their `lib/` and `settings.json` into `.clinerules/hooks/` for a project install (commit it to share with your team) or `~/Documents/Cline/Rules/Hooks/` for a global install.

## Step 3: Enable hooks in Cline

The final step is to turn hooks on inside Cline: **Settings → Features → Hooks**. Until hooks are enabled, the scripts are installed but Cline won't run them.

## How the hooks use memory

Cline exposes lifecycle hooks, and the integration works at four of them:

| Cline hook         | What Hindsight does                                                   |
| ------------------ | --------------------------------------------------------------------- |
| `TaskStart`        | Recall context for the new task and inject it.                        |
| `UserPromptSubmit` | Recall memories for your message; record the prompt for later retain. |
| `TaskComplete`     | Retain the task's transcript and summary.                             |
| `TaskCancel`       | Retain the partial transcript of a cancelled task.                    |

Recalled memories are injected as a `<hindsight_memories>` context block. Cline doesn't hand hooks a transcript, so the integration accumulates each task's prompts locally and retains them at task end. Memories land in a single bank (`cline` by default).

For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Configuration

Defaults live in the installed `settings.json`. Put personal overrides in `~/.hindsight/cline.json` (stable across reinstalls), or use `HINDSIGHT_*` environment variables.

Common keys include `hindsightApiUrl`, `hindsightApiToken`, `bankId`, `autoRecall`, `autoRetain`, `recallBudget`, `dynamicBankId`, and `debug`. For example, set `dynamicBankId` to `true` for a separate bank per project or session, or turn off `autoRetain` if you only want recall.

For the full set of configuration options, see the [package README](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/cline).

## Verify that memory is working

A good test sequence is:

1. run `hindsight-cline install` with your URL and key, and enable hooks in Cline
2. start a task and let Cline record a decision or convention
3. complete the task so the transcript is retained
4. start a new task in the same project
5. ask about the earlier decision

If the recalled `<hindsight_memories>` block surfaces the earlier decision, the setup is working. You can also check the `cline` bank via the API or dashboard to confirm a memory appeared.

## Common mistakes

### Forgetting to enable hooks in Cline

Installing the scripts is not enough. Turn hooks on under Settings → Features → Hooks, or the hooks never run.

### Running on Windows

Cline hooks run on macOS and Linux only, and require Python 3. On Windows the hooks won't execute.

### Testing retain too early

The transcript is retained when a task completes or is cancelled. If you check before the task ends, the session may not have been stored yet.

### Assuming a per-project bank by default

Memory lands in a single `cline` bank by default. If you want a separate bank per project or session, set `dynamicBankId`.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — install `hindsight-all`, run `hindsight-api`, and point the installer at it with `--api-url http://localhost:8888`.

### Does this use MCP?

No. The integration runs entirely on Cline's lifecycle hooks, so memory is deterministic and doesn't depend on the model deciding to call a tool.

### How is memory scoped?

Everything lands in a single `cline` bank by default. Set `dynamicBankId` to `true` for a separate bank per project or session.

### Which platforms are supported?

macOS and Linux only. Cline hooks do not run on Windows, and they require Python 3.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [Cline integration docs](https://hindsight.vectorize.io/docs/integrations/cline)
