---
title: "Cline Persistent Memory: Lifecycle Hooks Instead of MCP"
authors: [benfrank241]
slug: "2026/06/09/cline-persistent-memory"
date: 2026-06-09T12:00
tags: [cline, memory, persistent-memory, hindsight, coding-agents, tutorial]
description: "Add persistent memory to Cline with Hindsight using lifecycle hooks. No MCP server, no model tool-calling required. Hooks deterministically recall context before each task and retain what happened after."
image: /img/blog/cline-persistent-memory.png
hide_table_of_contents: true
---

![Cline Persistent Memory with Hindsight](/img/blog/cline-persistent-memory.png)

[Cline](https://github.com/cline/cline) is one of the most-used AI coding agents in VS Code. It reads files, runs commands, writes code, and iterates on a task until it's done. What it doesn't do is remember anything once a task ends. The next task opens cold, with no recollection of decisions you made, conventions you've established, or fragile areas of the codebase you've found the hard way.

This post is a walkthrough of the new Hindsight + Cline integration. It uses **Cline's lifecycle hooks** to add automatic recall before each task and automatic retain when a task ends, with no MCP server in the loop, and the model doesn't have to decide to call a memory tool. Memory just happens.

## TL;DR

<!-- truncate -->

- Cline has no persistent memory built in. Tasks restart from zero each time.
- The Hindsight integration installs four lifecycle hook scripts (`TaskStart`, `UserPromptSubmit`, `TaskComplete`, `TaskCancel`) plus a small Python lib. `pip install hindsight-cline`, one `install` command, and the hooks themselves run on stdlib Python 3 — no runtime dependencies.
- **Recall is deterministic.** Because it runs on hooks, memory is injected automatically. There's no MCP tool the model can forget to use.
- Recalled memories appear inside Cline as a `<hindsight_memories>` block, scoped to the current task description and your in-progress prompt.
- Hindsight Cloud means no local daemon. Memory is stored server-side and follows you across machines. [Sign up free.](https://ui.hindsight.vectorize.io/signup)
- **Platform note:** Cline hooks run on **macOS and Linux** only, with no Windows support.

## The Problem: Cline Has No Memory Between Tasks

Cline is fast and capable inside a single task. It can read your whole project, edit dozens of files, run tests, and converge on a solution. But when the task ends, everything it learned is gone. The next task starts with the model's training data plus whatever files Cline reads, with nothing you taught it before.

For one-off tasks that's fine. For an agent you use every day on the same codebase, it's a problem. You re-explain the same conventions, re-warn it about the same pitfalls, and re-state the same architectural decisions every time. The agent never gets to know your codebase the way a teammate would.

Hindsight closes that gap by giving Cline a persistent memory bank, and the lifecycle-hook integration wires it in without any in-task tool calls.

## How Hindsight Adds Memory to Cline

Cline supports [lifecycle hooks](https://docs.cline.bot/customization/hooks): small executable scripts it runs at key moments. The Hindsight integration installs four of them and routes each event to a Hindsight API call:

| Cline hook         | What Hindsight does                                                   |
| ------------------ | --------------------------------------------------------------------- |
| `TaskStart`        | Recall context for the new task description; inject it.               |
| `UserPromptSubmit` | Recall memories for your message; record the prompt for retain later. |
| `TaskComplete`     | Retain the task's accumulated transcript and final summary.           |
| `TaskCancel`       | Retain the partial transcript of a cancelled task.                    |

Because it runs on hooks, memory is **deterministic**. There's no MCP tool the model can forget to call and no extra latency from a tool-use round-trip. The recall and retain logic runs at well-defined points in Cline's task lifecycle.

```
Task starts ─ TaskStart ─────────► recall(task description) → inject memories
You send a message ─ UserPromptSubmit ─► recall(prompt) → inject memories
                                          (and append the prompt to the task transcript)
Task completes ─ TaskComplete ──► retain(accumulated transcript + summary)
Task cancelled ─ TaskCancel ────► retain(partial transcript)
```

One Cline-specific detail worth knowing: **Cline doesn't hand hooks a transcript.** Each hook gets the task ID and the current event payload, not the running conversation. The integration accumulates each task's prompts in `~/.hindsight/cline/state/` as it goes, and the end-of-task hook reads that back to retain the full transcript at once. The model never sees this bookkeeping; it just sees memories show up in context when relevant.

## Installing

The installer is a small CLI that copies the four hook files (plus their shared lib and a `settings.json`) into Cline's hooks directory. Install it with pip:

```bash
pip install hindsight-cline
```

Then, from your project directory:

```bash
hindsight-cline install \
  --api-url https://api.hindsight.vectorize.io \
  --api-token YOUR_KEY
```

That installs to `.clinerules/hooks/`; commit it to share with your team. To install globally (apply to every project), add `--global`:

```bash
hindsight-cline install --global \
  --api-url https://api.hindsight.vectorize.io \
  --api-token YOUR_KEY
```

This drops hooks into `~/Documents/Cline/Rules/Hooks/` instead. (`hindsight-cline uninstall` removes them.)

**Final step, enable hooks in Cline:** Settings → Features → Hooks (toggle on).

Cline hooks run on **macOS and Linux only**. They use Python 3 (any modern system Python works, with no `pip install` needed at runtime).

## Hindsight Cloud (Recommended)

The fastest path is Hindsight Cloud: no daemon to keep alive, memory syncs across machines, and the extraction work happens server-side. That matters more for Cline than for a server-side agent. Cline lives in VS Code, which most developers use across a laptop, a desktop, and sometimes a remote dev box; Cloud means the same memory bank shows up everywhere without copying files around. Because extraction runs server-side, you also don't have to thread an LLM API key into the hook environment (the retain hook would otherwise need one to call the extraction model), and there's no `hindsight-api` process you have to remember to start before opening VS Code. The installer's `--api-url` and `--api-token` flags configure it in one step. Your connection settings land in `~/.hindsight/cline.json`, which is stable across reinstalls:

```json
{
  "hindsightApiUrl": "https://api.hindsight.vectorize.io",
  "hindsightApiToken": "hsk_your_token"
}
```

Create an account and grab an API key at [hindsight.vectorize.io](https://ui.hindsight.vectorize.io/signup).

Self-hosting works exactly the same way: start the API locally and point the installer at it:

```bash
pip install hindsight-all
export HINDSIGHT_API_LLM_API_KEY=your-openai-key
hindsight-api  # http://localhost:8888
```

Then re-run the installer with `--api-url http://localhost:8888`.

## What Gets Recalled

`TaskStart` and `UserPromptSubmit` both run a Hindsight recall and return a `<hindsight_memories>` block as context that Cline injects before the model sees your prompt:

```
<hindsight_memories>
Relevant memories from past conversations. Only use memories that are directly useful to continue this task; ignore the rest:
Current time - 2026-06-09 13:42

- Project uses asyncpg, not SQLAlchemy; switched after Redis cache stampede in March [world]
- Tests live under tests/integration/ and run via `make test-int`, not pytest directly [world]
- The `auth_v2` module is being deprecated; new code should target `identity/` [experience]
</hindsight_memories>
```

Cline sees this block; it doesn't appear in your editor output. The result: Cline starts every task with relevant past context already in scope, without you having to provide it.

You can tune how much context to pull with `recallBudget` (`"low"` / `"mid"` / `"high"`) and `recallMaxTokens`.

### Before and after

Without persistent memory, a new task in Cline starts cold. You type "fix the broken auth tests" and Cline reads the test file, makes reasonable guesses about which auth module is in scope, and may try patterns you already rejected.

With Hindsight, the same task opens with recalled context: that `auth_v2` is deprecated, that the test runner is `make test-int`, that the recent fix-stack landed in `identity/`. Cline picks the right module and the right test command on the first turn instead of the third.

## Per-Project Memory

By default all Cline tasks share a single bank (`cline`). To give each project its own isolated bank, switch to dynamic bank IDs in `~/.hindsight/cline.json`:

```json
{
  "dynamicBankId": true,
  "dynamicBankGranularity": ["agent", "project"]
}
```

Bank IDs are derived from the workspace path (`agent::project`), so a task in `~/projects/api` writes to a different bank than one in `~/projects/frontend`. Switching folders automatically switches memory context.

Valid granularity fields are `agent`, `project`, `session`, and `user`. Adding `user` (sourced from the `HINDSIGHT_USER_ID` env var) is useful if multiple people share a machine but should not share recall.

## Team Shared Memory

Individual persistent memory is useful. Shared memory across a team is transformative.

When everyone on a team points their Cline config at the same Hindsight bank, context accumulated by one developer becomes available to all. A bug discovered on Monday surfaces in recall on Tuesday, regardless of who's asking. Architecture decisions made in one task inform the next, without requiring anyone to update a shared doc.

To configure team shared memory, set a fixed `bankId` in each developer's config and point them at the same Hindsight Cloud endpoint:

```json
{
  "hindsightApiUrl": "https://api.hindsight.vectorize.io",
  "hindsightApiToken": "hsk_your_token",
  "bankId": "my-team-project"
}
```

See [Shared Memory for AI Coding Agents](https://hindsight.vectorize.io/blog/2026/03/31/team-shared-memory-ai-coding-agents) for a full team setup guide.

## Key Configuration Options

Settings live in `~/.hindsight/cline.json` (personal overrides) or the installed `settings.json` (defaults). Every setting can also be set via `HINDSIGHT_*` environment variables.

| Setting         | Default                  | What it does                                            |
| --------------- | ------------------------ | ------------------------------------------------------- |
| `bankId`        | `cline`                  | Memory bank for this integration.                       |
| `autoRecall`    | `true`                   | Inject memories before tasks/prompts.                   |
| `autoRetain`    | `true`                   | Retain the task transcript when it ends.                |
| `recallBudget`  | `mid`                    | Recall depth: `low` (fast) / `mid` / `high` (thorough). |
| `recallTypes`   | `["world","experience"]` | Memory categories to recall.                            |
| `retainMission` | generic                  | Steers fact extraction; tell it what to focus on.       |
| `dynamicBankId` | `false`                  | Per-project bank isolation.                             |
| `debug`         | `false`                  | Log activity to stderr.                                 |

A focused `retainMission` makes the extracted memories meaningfully better:

```json
{
  "retainMission": "Extract technical decisions, code patterns, debugging solutions, user preferences, project context, and architectural choices. Ignore routine greetings and transient operational details."
}
```

## Pitfalls

**Hooks not firing.** The installer copies the files in, but the toggle is still off by default. Go to Settings → Features → Hooks in Cline and turn it on. A quick way to verify hooks run: enable `debug: true` and watch stderr for `[Hindsight]` lines.

**No memories recalled in the first task.** Recall only returns results after something has been retained. Complete one real task first; the second one starts seeing recalled context.

**Nothing happening on Windows.** Cline's hook runner is macOS/Linux only; there is currently no Windows path. (If you're on Windows and want persistent memory for a coding agent, the [Hermes + Hindsight](https://hindsight.vectorize.io/blog/2026/06/01/hermes-hindsight-windows-setup) setup is a good alternative.)

**Smoke-testing a hook without Cline.** You can pipe a synthetic event into a hook script directly to make sure it works end-to-end:

```bash
echo '{"hookName":"UserPromptSubmit","prompt":"how do we authenticate?","taskId":"t1","workspaceRoots":["/tmp/x"]}' \
  | .clinerules/hooks/UserPromptSubmit
# → {"cancel": false, "contextModification": "<hindsight_memories>…", "errorMessage": ""}
```

## Tradeoffs

**Recall adds latency.** Every prompt triggers a Hindsight query before Cline sees it. With Hindsight Cloud and a fast connection that's typically under 300ms, imperceptible in interactive use. Drop `recallBudget` to `"low"`, or set `autoRecall: false`, if you need to skip it.

**Retain runs at task end, not mid-task.** Memories from the task you're in become available _after_ it completes. If you cancel a task you'd otherwise want to recall from, the `TaskCancel` hook still retains the partial transcript, but you have to actually cancel to trigger it.

**Extraction quality depends on conversation quality.** Hindsight extracts facts from what's in the transcript. If a task is all file edits and no narration, there's little for the extractor to work with. A few sentences explaining what you decided and why go a long way.

## Recap

|                           | Cline default               | With Hindsight                            |
| ------------------------- | --------------------------- | ----------------------------------------- |
| Memory across tasks       | None                        | Automatic                                 |
| Memory setup              | Manual `.clinerules` / docs | Extracted from task transcripts           |
| Recall mechanism          | Files Cline reads each task | Semantic search, injected per task/prompt |
| Per-project isolation     | No                          | Optional via `dynamicBankId`              |
| Team shared memory        | No                          | Shared bank via Hindsight Cloud           |
| Model tool-calling needed | n/a                         | No (lifecycle hooks)                      |

## Next Steps

- **Hindsight Cloud:** [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io/signup)
- **Integration docs:** [Cline + Hindsight](/sdks/integrations/cline)
- **Source:** [vectorize-io/hindsight/hindsight-integrations/cline](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/cline)
- **Team memory:** [Shared Memory for AI Coding Agents](https://hindsight.vectorize.io/blog/2026/03/31/team-shared-memory-ai-coding-agents)
- **Windows alternative:** [Hermes Agent with Hindsight](https://hindsight.vectorize.io/blog/2026/06/01/hermes-hindsight-windows-setup)
