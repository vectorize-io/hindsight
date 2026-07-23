---
sidebar_position: 6
title: "Coding Agents Memory Plugin (opencode, Claude Code, Codex, Cursor) | Integration Guide"
description: "One Hindsight memory plugin for coding agents: per-repo memory banks from git history and past conversations, injected into opencode, Claude Code, Codex CLI, and Cursor CLI at the moment the agent starts working."
---

# Coding Agents

Long-term **project memory** for coding agents, from one package: a shared reflect-and-inject core
with a thin entry point per agent — **opencode**, **Claude Code**, **Codex CLI**, **Cursor CLI** —
plus a one-shot **backfill** CLI that ingests a repo's git history and past developer conversations
into a [Hindsight](https://vectorize.io/hindsight) memory bank.

The premise: most of a real fix is derivable from the code, but the *last mile* often hinges on a
project-specific decision that isn't in the code at all — a rounding rule, a retry allowlist, a
tie-break policy. Those decisions live in git history and past conversations. This plugin puts them
in front of the agent at the moment it starts working.

## How it works

1. **Backfill (once per repo).** `hindsight-coding-backfill --repo .` ingests every git commit
   (message + diff) and, optionally, past developer conversations, under two retain strategies tuned
   per content type (`git`: verbose decision extraction; `chat`: at most two coherent facts per
   conversation — the final decision and the notable rejected alternative). It then synthesizes
   **knowledge pages** (a codebase mental map) and tags every item with a `REF-ID` so any surfaced
   fact traces back to its commit or session.
2. **Reflect once per session.** On the session's first task message, the entry point sends that
   message to Hindsight `reflect`, which reasons over the bank and returns a synthesized
   **root-cause answer** — the exact rule and literal values that were decided, with citations.
3. **Inject every turn.** The answer is pushed into the agent's context (system prompt on opencode;
   hook context on Claude/Codex/Cursor, cached per session and re-injected on later prompts) so it
   survives long sessions and correction rounds.
4. **Never break the agent — never fail silently.** A failed reflect degrades to no-memory, but
   every outcome (`reflect_ok` / `reflect_empty` / `reflect_failed`, with duration and error) is
   appended to a diagnostics file, so a memory-less session can't masquerade as a memory session.

When memories **conflict** on the same rule, reflect prefers the latest/superseding decision — a
rule amended in a later conversation wins over the original, and the superseded rule is reported as
no longer in effect.

## Supported agents

| harness       | kind              | entry point             | install |
| ------------- | ----------------- | ----------------------- | ------- |
| `opencode`    | persistent plugin | package default export  | add the package dir to `opencode.json` → `"plugin": [...]` |
| `claude-code` | per-prompt hook   | `hindsight-claude-hook` | `UserPromptSubmit` hook in Claude Code `settings.json` |
| `codex`       | per-prompt hook   | `hindsight-codex-hook`  | `UserPromptSubmit` hook in `~/.codex/hooks.json` (+ `codex_hooks = true`, Codex CLI ≥ 0.116) |
| `cursor-cli`  | per-prompt hook   | `hindsight-cursor-hook` | `beforeSubmitPrompt` hook in Cursor `hooks.json` |

```json title="opencode.json"
{ "plugin": ["/path/to/hindsight-coding-agents"] }
```

```json title="Claude Code settings.json — Codex ~/.codex/hooks.json is identical (command: hindsight-codex-hook)"
{ "hooks": { "UserPromptSubmit": [ { "hooks": [
    { "type": "command", "command": "hindsight-claude-hook" } ] } ] } }
```

```json title="Cursor hooks.json"
{ "hooks": { "beforeSubmitPrompt": [ { "command": "hindsight-cursor-hook" } ] } }
```

The opencode plugin additionally exposes an on-demand **`memory_reflect` tool** (same synthesized
reflect, callable mid-task) and opt-in **incremental git-sync** and **live session write-back**
(see the configuration reference).

## Configuration

All configuration is **JSON files, no environment variables** (exception: `HINDSIGHT_DIAG_FILE` for
the diagnostics path). Layering, later wins per field:

1. built-in defaults
2. `~/.hindsight/coding-agent.json` — user-global
3. its `harnesses.<name>` section — per-agent override
4. the **nearest** `<dir>/.hindsight/coding-agent.json` at or above the working directory —
   project-local (the natural home for per-repo settings)
5. its `harnesses.<name>` section

Each entry point knows which harness it *is*, so one shared config serves several agents side by
side:

```jsonc
{
  "apiUrl": "http://localhost:8888",
  "harnesses": {
    "opencode":    { "reflectTimeoutMs": 60000 },
    "claude-code": { "disabled": true }          // e.g. memory off for Claude only
  }
}
```

### Reference

| field | default | meaning |
| --- | --- | --- |
| `apiUrl` | `http://localhost:8888` | Hindsight API base URL |
| `apiToken` | — | bearer token (Hindsight Cloud) |
| `bankId` | — | **explicit static bank**; unset ⇒ per-repo dynamic resolution (below) |
| `dynamicBankId` | dynamic iff no `bankId` | force dynamic (`true`) or static (`false`) resolution |
| `bankIdTemplate` | `"{gitProject}"` | dynamic bank id format, e.g. `"hindsight-{gitProject}"` |
| `directoryBankMap` | — | absolute path → bank; **longest prefix wins**; overrides everything |
| `resolveWorktrees` | `true` | `{gitProject}`: linked worktrees share the main repo's bank |
| `disabled` | `false` | hard off-switch (inert plugin/hook — a no-memory baseline) |
| `reflectTimeoutMs` | `120000` | reflect timeout; on timeout the agent runs without memory (recorded in diagnostics) |
| `retainSessions` | `false` | opencode only: upsert the live transcript into the bank every N turns |
| `retainEveryTurns` | `5` | write-back cadence (user turns) |
| `gitSync.enabled` | `false` | opencode only: on load, retain commits new since the backfill |
| `gitSync.ref` | `origin/main` | git-sync target ref (falls back to `HEAD`) |
| `gitSync.fetch` | `false` | `git fetch` the ref before diffing |
| `harnesses.<name>` | — | per-harness override of any field above |
| `harness` | `opencode` | **backfill only**: which session format `--conversations` is read as |

### Bank resolution — per-repo by default

Coding memory is **per repository**. Resolution order for the working directory:

1. `directoryBankMap` — longest matching absolute-path prefix (mapping a repo root covers every
   subdirectory; deeper mappings win; overrides even an explicit `bankId`).
2. Static — `bankId` set (or `dynamicBankId: false`).
3. Dynamic — `bankIdTemplate` with placeholders:
   - `{gitProject}` — worktree-aware repo name: every linked worktree resolves to the **main**
     worktree's basename, so all worktrees of a repo share one bank (bare repos use the bare dir
     name; non-git directories fall back to the dir basename)
   - `{project}` — plain working-directory basename
   - `{harness}` — the entry point asking (`opencode`, `claude-code`, `codex`, `cursor-cli`)
   - `{channel}` / `{user}` — `$HINDSIGHT_CHANNEL_ID` / `$HINDSIGHT_USER_ID`

The default `"{gitProject}"` means **all agents share one memory per repo** — use
`"{harness}-{gitProject}"` to split per agent instead.

## Backfill

```bash
hindsight-coding-backfill --repo /path/to/repo \
  [--bank myproject] [--harness opencode] [--conversations sessions.json] \
  [--api-url http://localhost:8888] [--api-token X] [--config <path>] \
  [--limit 100] [--reset] [--no-pages] [--concurrency 8]
```

- Without `--bank`, the **same per-repo resolution** the runtime uses is applied to `--repo`, so
  `hindsight-coding-backfill --repo .` fills exactly the bank the agents will read.
- `sessions.json` is a normalized interchange format any exporter can emit:
  `[{ "id": "s1", "turns": [{ "role": "user", "text": "...", "timestamp?": "ISO" }, ...] }, ...]`.
  Session list order is **chronological** — a later chat can amend an earlier one, and recency
  follows list order (last = newest).
- Tip: validate a setup with `--limit 100` before a full-history ingest.

## Diagnostics

Every reflect outcome is appended as a JSON line to `/tmp/hindsight-plugin.log` (override with
`HINDSIGHT_DIAG_FILE`):

```json
{"ts":"2026-07-23T07:05:52Z","harness":"claude-code","event":"reflect_ok","ms":15816,"chars":324,"query":"..."}
```

`reflect_failed` records the error; if you're comparing memory-on vs memory-off, check this file —
a run whose reflects failed is a no-memory run.
