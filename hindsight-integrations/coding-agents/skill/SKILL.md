---
name: hindsight-coding-agent
description: How this machine's Hindsight coding-agent memory works — the plugin behind the 🧠 banner. Use when the user says "store/remember this in hindsight", asks what the memory/knowledge pages are, wants to configure per-repo memory (disable, rename banks, git depth), or something memory-related looks broken.
---

# Hindsight Coding-Agent Memory

This machine runs the `hindsight-coding-agents` plugin: long-term project memory for coding
sessions, backed by a Hindsight server. You (the agent) are already wired into it — this skill
explains what happens automatically, which tools you have, and how to configure or debug it.

## What happens automatically (no action needed)

- **Per-repo memory bank**: each repository resolves to a bank (shown in the session banner:
  `↳ memory bank “coding-agent::<repo>”`). Worktrees share the main repo's bank.
- **Ingestion builds itself**: on first open, the bank is seeded from recent commit messages and a
  read-only codebase survey; every session start, a background engine tops it up (new commits, new
  conversations) and keeps 5 knowledge pages current. There is NO ingest command to run.
- **Session synthesis**: the first prompt of a session triggers one deep memory synthesis
  (`reflect`) injected into context. Later turns inject nothing automatically.
- **Write-back**: the session transcript is retained into the bank automatically at session end
  (per-turn on opencode). The user never needs to "save" a conversation.

## Storing things deliberately

When the user says "store this in hindsight" / "remember this":

- The **current conversation** is captured automatically at session end — say so; no tool needed.
- An **external document, notes, or durable findings** → `hindsight_ingest_document(title, content)`.
- A **new feature/initiative being started** → `hindsight_capture_initiative(title, summary)`,
  once, right after the plan is agreed and before code is written.

## Retrieving

- `hindsight_search_knowledge_pages(query)` — FIRST STOP for project questions (components,
  conventions, past decisions, initiatives). Server-side hybrid search, fast.
- `hindsight_read_knowledge_page(page_id)` / `hindsight_list_knowledge_pages` — read pages fully.
- `hindsight_reflect(query)` — deep reasoning over the whole memory for WHY questions and exact
  decided values; slower (seconds), use deliberately.
- Credit visibly whenever memory informs an answer: start that part with
  `🧠 From Hindsight memory (<page>): …` — and never credit memory that didn't contribute.

## Correcting wrong or stale memory

If you verify that something Hindsight served is wrong or outdated (the code, git, or an external
source contradicts it), FIX THE RECORD — don't just ignore it. Call
`hindsight_ingest_document` with:

- **title**: `Correction: <topic>` (e.g. `Correction: retry policy 4xx set`)
- **content**: (1) what memory claimed, (2) what is verifiably true now, (3) the evidence you
  checked (file/commit/output). Quote exact values verbatim.

Newer facts supersede older ones in retrieval, so one clear correction permanently outranks the
stale memory. Do this whenever you catch a wrong injected memory, a stale knowledge-page claim, or
an outdated decision — silent disregard leaves the trap armed for the next session.

## Configuration — ONE file: `~/.hindsight/coding-agent.json`

No environment variables (exceptions: `HINDSIGHT_CONFIG` relocates this file;
`HINDSIGHT_DIAG_FILE`/`HINDSIGHT_LOG_FILE`/`HINDSIGHT_LOG_LEVEL` for diagnostics).
Layering, later wins: defaults → file → `harnesses.<name>` → `banks.<resolvedBankId>`.

```jsonc
{
  "apiUrl": "http://localhost:8888", // your Hindsight server
  "apiToken": "…", // Hindsight Cloud only
  "gitIngest": "message", // "message" | "full" (per-commit diffs) | "none"
  "harnesses": { "claude-code": { "disabled": true } }, // per-agent override of anything
  "mapPathToBank": { "/Users/me/work/client-x": "client-x-memory" }, // path-prefix → bank
  "banks": {
    // per-repo control, keyed by RESOLVED bank id
    "coding-agent::secret": { "disabled": true }, // blacklist a repo
    "coding-agent::old": { "bank": "team::shared" }, // rename / converge banks (single hop)
    "coding-agent::mono": { "gitIngest": "full", "retainSessions": false },
  },
}
```

Key behavioral fields (any of them valid per-harness or per-bank): `disabled`,
`retainSessions` (write-back opt-out), `gitIngest`, `reflectTimeoutMs` (default 120000; hooks cap
at 25s), `autoReflect` (true; false = no injected first-prompt synthesis — the agent is instead
told to call `hindsight_reflect` on new goals), `pageRefreshEveryTurns` (10),
`pageTriggerType`/`pageTriggerCron` (when NEW knowledge pages refresh: `auto-refresh` (default) after
each consolidation, `cron` on a schedule, `manual` never — existing pages keep the trigger they were
created with), `autoSeed`/`seedLimit` (true/300),
`codebaseSurvey`/`surveyModel`/`surveyBudgetUsd` (true/haiku/2), `surveyRefreshCommits` (0=off),
`logLevel` ("info").

Blacklist a whole directory tree: map it to one bank and disable that bank —
`"mapPathToBank": {"~/scratch": "scratch"}` + `"banks": {"scratch": {"disabled": true}}`.

Bank resolution order: `mapPathToBank` longest prefix → static `bankId` → template
(default `coding-agent::{gitProject}`) → the matching `banks.<id>` section (its `bank` field
renames the destination). Two repos share memory by converging their `banks.<id>.bank` on one
name, or by one `mapPathToBank` prefix over their parent directory.

## Install / update (for setting up another machine or harness)

```bash
npx @vectorize-io/hindsight-coding-agents install all     # every detected agent
npx @vectorize-io/hindsight-coding-agents install codex   # or specific: opencode|claude-code|codex|antigravity-cli|cursor-cli
npx @vectorize-io/hindsight-coding-agents uninstall       # removes exactly what install added
# updating is the same install command again — it re-copies the runtime in place
```

## Debugging

- **Readiness**: `hindsight_sync_status` — `synced: true` = seeded memory queryable; also shows
  gitlog freshness, per-commit deepening progress, survey state (`surveyDocs` 0–4 = findings
  present; baseline without findings retries automatically), and active extraction ops.
- **Logs**: `$TMPDIR/hindsight-coding-agent/plugin.log` (leveled; set `"logLevel": "debug"` or
  `HINDSIGHT_LOG_LEVEL=debug` to mirror every event) and `/tmp/hindsight-plugin.log` (structured
  JSONL diag events with timings: `session_start`, `reflect_ok`, `deepen_done`, `retain_ok`, …).
- **Reset a repo's memory**: delete its bank on the server — the bank is the ONLY state; the next
  session is a true first-open. No client files to clean.
- **Rule of thumb**: memory silently missing → check the diag log for whether `session_start`/
  `deepen_started` ever fired for that bank; a session started before the plugin was installed has
  no SessionStart behind it (its first prompt after install self-heals).
- **Internal marker docs you may notice** (safe to ignore, safe to delete): `survey-baseline:<sha>`
  — "🛰️ researching…" while a codebase survey runs, flipped to "✅ completed" once its findings
  land. Retained under the `survey` strategy, whose marker rule extracts NOTHING from status markers;
  powers the re-survey cadence and `surveyBaseline` in sync status. `gitlog:<repo>` is the
  aggregated commit-message seed document.
- Failures never break the agent: reflect/pages/retain failures degrade to a normal memoryless
  turn and are recorded in the logs.
