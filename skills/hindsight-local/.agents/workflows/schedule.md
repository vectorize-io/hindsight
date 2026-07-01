---
name: schedule
description: Register a scheduled agent job from a natural-language schedule request — parse the interval, resolve agent-id + prompt + workspace, call oma schedule:add, then confirm with oma schedule:list
disable-model-invocation: true
---

# MANDATORY RULES: VIOLATION IS FORBIDDEN

- **Response language follows `language` setting in `.agents/oma-config.yaml` if configured.**
- **NEVER skip steps.** Execute from Step 1 in order.
- **This workflow is slash-invoked only** (`/schedule`). It is NOT triggered by broad keyword detection.

---

## Step 1: Collect Schedule Request

Ask the user for the following if not already provided in the prompt:

| Field | Description | Example |
|-------|-------------|---------|
| Agent ID | Identifier of the agent to run | `qa-reviewer`, `backend-engineer` |
| Prompt | Instruction the agent will receive | `"review the latest diff"` |
| Interval / cron | When to run | `"every 2 hours"`, `"5m"`, `"0 9 * * *"` |
| Workspace (optional) | Absolute path to the project directory | `/home/user/myproject` (default: cwd) |
| Vendor (optional) | CLI vendor override (passed to `oma agent:spawn -m`) | `claude`, `codex`, `antigravity`, `cursor`, `qwen`, `grok`, `opencode` |

If all required fields are already in the user's prompt, proceed directly to Step 2.

---

## Step 2: Resolve Interval Format

Determine whether the user's schedule phrase is:

- **Natural-language interval** — use `--every "<phrase>"` (e.g. `"every 2 hours"`, `"5m"`, `"every 30 minutes"`)
- **5-field cron expression** — use `--cron "<expr>"` (e.g. `"0 9 * * *"`)

If the phrase is ambiguous (e.g. "twice a day", "weekdays at 9am"), ask the user to clarify or suggest the closest cron equivalent and confirm before proceeding.

---

## Step 3: Register the Job

Run the appropriate `oma schedule:add` command:

```bash
# Natural-language interval
oma schedule:add <agent-id> "<prompt>" --every "<phrase>" [--model <vendor>] [--workspace <path>] [--once]

# Explicit cron expression
oma schedule:add <agent-id> "<prompt>" --cron "<expr>" [--model <vendor>] [--workspace <path>] [--once]
```

If the interval was rounded (the CLI prints a "Note:" line), surface that note to the user and ask for confirmation before continuing.

---

## Step 4: Confirm Registration

Run `oma schedule:list` to display all registered jobs and confirm the new job appears:

```bash
oma schedule:list
```

Show the output to the user. Verify the new job is listed with `drift: synced`.

If `drift: missing-in-os` is shown, suggest running `oma schedule:sync`.

---

## Step 5: Summarise

Report to the user:

- Agent ID and prompt registered
- Cron expression used (and the original interval phrase if `--every` was used)
- Workspace and vendor
- Whether the job is recurring or one-shot (`--once`)
- Next suggested action if the job did not sync
