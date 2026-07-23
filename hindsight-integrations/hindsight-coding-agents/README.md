# hindsight-coding-agents

Long-term project memory for **coding agents**, backed by [Hindsight](https://vectorize.io/hindsight).
One package, several agents: a shared reflect-and-inject core with a thin entry point per agent
(**opencode**, **Claude Code**, **Codex CLI**, **Cursor CLI**), plus a one-shot **backfill** CLI that
ingests a repo's git history and past developer conversations into a memory bank.

The premise: most of a real fix is derivable from the code, but the *last mile* often hinges on a
project-specific decision that isn't in the code at all — a rounding rule, a retry allowlist, a
tie-break policy. Those decisions live in git history and past conversations. This package puts them
in front of the agent at the moment it starts working.

## How it works

1. **Backfill (once per repo).** `hindsight-coding-backfill --repo .` ingests every git commit
   (message + diff) and, optionally, past developer conversations into the repo's bank, under two
   retain strategies tuned per content type (`git`: verbose decision extraction; `chat`: at most two
   coherent facts per conversation — the final decision and the notable rejected alternative). It
   then synthesizes **knowledge pages** (a codebase mental map) and tags every item with a
   `REF-ID` so any surfaced fact traces back to its commit or session.
2. **Reflect once per session.** On the session's first task message, the entry point sends that
   message to Hindsight `reflect`, which reasons over the bank and returns a synthesized
   **root-cause answer** — the exact rule and literal values that were decided, with citations.
3. **Inject every turn.** The answer is pushed into the agent's context (system prompt on opencode;
   hook context on Claude/Codex/Cursor, cached per session and re-injected on later prompts) so it
   survives long sessions and correction rounds.
4. **Never break the agent — never fail silently.** A failed reflect degrades to no-memory, but every
   outcome (`reflect_ok` / `reflect_empty` / `reflect_failed`, with duration and error) is appended to
   a diagnostics file, so a memory-less session can't masquerade as a memory session.

When memories **conflict** on the same rule, reflect prefers the latest/superseding decision — a rule
amended in a later conversation wins over the original, and the superseded rule is reported as no
longer in effect.

## Harnesses

| harness       | kind              | entry point                                     | install |
| ------------- | ----------------- | ----------------------------------------------- | ------- |
| `opencode`    | persistent plugin | package default export (`dist/index.js`)        | add the package dir to `opencode.json` → `"plugin": [...]` |
| `claude-code` | per-prompt hook   | `hindsight-claude-hook` (`dist/claude-hook.js`) | `UserPromptSubmit` hook in Claude Code `settings.json` |
| `codex`       | per-prompt hook   | `hindsight-codex-hook` (`dist/codex-hook.js`)   | `UserPromptSubmit` hook in `~/.codex/hooks.json` (+ `codex_hooks = true`, Codex CLI ≥ 0.116) |
| `cursor-cli`  | per-prompt hook   | `hindsight-cursor-hook` (`dist/cursor-hook.js`) | `beforeSubmitPrompt` hook in Cursor `hooks.json` |

Hook-based harnesses share one runtime (`src/core/hook.ts`); each is a ~25-line spec of its event
fields and output schema. The opencode plugin additionally exposes an on-demand **`memory_reflect`
tool** (same synthesized reflect, callable mid-task) and the opt-in **incremental git-sync** and
**live session write-back** described in the configuration reference.

Install snippets:

```json
{ "plugin": ["/path/to/hindsight-coding-agents"] }
```

```jsonc
// Claude Code settings.json  /  Codex ~/.codex/hooks.json (identical protocol)
{ "hooks": { "UserPromptSubmit": [ { "hooks": [
    { "type": "command", "command": "hindsight-claude-hook" } ] } ] } }   // or hindsight-codex-hook

// Cursor hooks.json
{ "hooks": { "beforeSubmitPrompt": [ { "command": "hindsight-cursor-hook" } ] } }
```

Adding an agent: hook-based → write a `HookSpec` entry point (see `src/cursor-hook.ts`) and register
a `hookAdapter` in `src/harness/registry.ts`; persistent-plugin → implement `HarnessAdapter`
(`src/core/types.ts`) fully.

## Configuration

All configuration is **JSON files, no environment variables** (exception: `HINDSIGHT_DIAG_FILE` for
the diagnostics path). Layering, later wins per field:

1. built-in defaults
2. `~/.hindsight/coding-agent.json` — user-global
3. its `harnesses.<name>` section — per-agent override
4. the **nearest** `<dir>/.hindsight/coding-agent.json` at or above the working directory —
   project-local (the natural home for per-repo settings)
5. its `harnesses.<name>` section

Each entry point knows which harness it *is* (the opencode plugin is loaded by opencode, the codex
hook by Codex...), so one shared config serves several agents side by side:

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

### Bank resolution

Coding memory is **per repository**. Resolution order for the working directory:

1. `directoryBankMap` — longest matching absolute-path prefix (mapping a repo root covers every
   subdirectory; deeper mappings win; overrides even an explicit `bankId`).
2. Static — `bankId` set (or `dynamicBankId: false`).
3. Dynamic — `bankIdTemplate` with placeholders:
   - `{gitProject}` — worktree-aware repo name: `git rev-parse --git-common-dir` resolves every
     linked worktree to the **main** worktree's basename, so all worktrees of a repo share one bank
     (bare repos use the bare dir name; non-git directories fall back to the dir basename)
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
- `sessions.json` is the normalized interchange format any exporter can emit:
  `[{ "id": "s1", "turns": [{ "role": "user", "text": "...", "timestamp?": "ISO" }, ...] }, ...]`.
  Session list order is **chronological** — a later chat can amend an earlier one, and recency
  follows list order (last = newest).
- Chats are ingested **before** the git flood so decisions aren't starved in the extraction queue;
  the CLI drains extraction and reports `done/failed` counts before exiting.
- Tip: validate a setup with `--limit 100` before a full-history ingest.

Local Hindsight for trying it out:

```bash
docker run -d -p 8888:8888 -p 9999:9999 -e HINDSIGHT_API_LLM_PROVIDER=gemini \
  -e HINDSIGHT_API_LLM_API_KEY=$GEMINI_API_KEY -e HINDSIGHT_API_LLM_MODEL=gemini-2.5-flash \
  ghcr.io/vectorize-io/hindsight:latest
```

## Diagnostics

Every reflect outcome is appended as a JSON line to `/tmp/hindsight-plugin.log` (override with
`HINDSIGHT_DIAG_FILE`):

```json
{"ts":"2026-07-23T07:05:52Z","harness":"claude-code","event":"reflect_ok","ms":15816,"chars":324,"query":"..."}
```

`reflect_failed` records the error; if you're comparing memory-on vs memory-off, check this file —
a run whose reflects failed is a no-memory run.

## Testing

```bash
npm test          # unit tests: bank resolution matrix + config layering (no network)
npm run test:live # LIVE system test against a real server + real LLM:
                  #   HINDSIGHT_API_URL=http://localhost:8888 npm run test:live
```

The live suite builds a real git repo with a decision planted in a commit and a conversation, runs
the real backfill (server-side LLM extraction), then drives the built hook binaries as subprocesses
and asserts the decision's literal values come back in the injected context.

## Layout

```
src/
  core/          # harness-agnostic: config (layered), bank resolution, hindsight client, missions,
                 # git + chat ingest, git-sync, inject, hook runtime, RuntimeCore
  harness/       # per-agent adapters + registry (opencode persistent; claude/codex/cursor as hooks)
  index.ts       # opencode plugin entrypoint
  claude-hook.ts # Claude Code hook entrypoint    (bin: hindsight-claude-hook)
  codex-hook.ts  # Codex CLI hook entrypoint      (bin: hindsight-codex-hook)
  cursor-hook.ts # Cursor CLI hook entrypoint     (bin: hindsight-cursor-hook)
  backfill.ts    # backfill CLI                   (bin: hindsight-coding-backfill)
```
