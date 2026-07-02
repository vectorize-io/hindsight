# hindsight-coding-agents

Reflect-only [Hindsight](https://vectorize.io/hindsight) long-term memory for **coding agents**, plus a
one-shot **backfill** that ingests a repo's git history and past developer conversations into a
Hindsight bank. Harness-pluggable: the memory logic is shared, and a thin per-agent adapter binds it
to a specific coding agent (opencode today; others slot in via the harness registry).

## What it does

- **Runtime plugin (reflect + INJECT)** — on the first task message it phrases the bug's symptom to
  Hindsight `reflect`, gets back a synthesized **root-cause** answer drawn from the ingested history,
  and pushes it into the **system prompt** every turn (so it survives interventions). No tools, no
  recall — the memory is injected, not requested.
- **On-demand memory tool (`memory_reflect`)** — the agent can also query project memory itself, at any
  point mid-task, via a `memory_reflect` tool. Same synthesized `reflect` as the auto-injection, but on
  demand: give it a symptom/question and it returns the root-cause answer (exact rules/values + REF-ID
  citations) from this repo's history. Complements the automatic first-message injection.
- **Incremental git-sync (opt-in)** — when enabled (`gitSync.enabled: true`), on load the plugin checks
  whether the bank is up to date with the repo's target ref (`origin/main`, falling back to `HEAD`): it
  diffs the ref's commits against the commit `document_id`s already ingested and **async-retains only the
  missing ones**, using the same per-commit encoding as the backfill. Set-based, so it's correct across
  rebases/force-push; best-effort and non-blocking. Off by default.
- **Live write-back (opt-in)** — with `HINDSIGHT_RETAIN_SESSIONS` on, every few turns it upserts the
  user/assistant transcript (tool calls dropped) under a stable per-session `document_id`, so future
  sessions can learn from this one. Off by default.
- **Backfill command** — `hindsight-coding-backfill`:
  1. configures the bank: reflect mission, observations **on**, and two named **retain strategies**
     — `git` and `chat`;
  2. ingests **every** git commit (full message + full diff, no pre-filtering) under `git`;
  3. reads the chosen **harness**'s past sessions and ingests each **raw** as a JSON user/assistant
     transcript (custom extraction → ≤2 coherent facts) under `chat`;
  4. synthesizes generic **knowledge pages** (a codebase mental map) from the ingested memory;
  5. tags every item with a `REF-ID` so a reflected fact traces back to its commit/session.

Git and chat use **different Hindsight retain strategies** (per-item `strategy`), so each content
type is extracted with settings suited to it — in one bank, one pass.

## Harnesses

A "harness" is a coding agent. Each one differs in exactly two places — everything else is shared
core (`src/core/`):

| harness    | past-session source (backfill)             | runtime binding             |
| ---------- | ------------------------------------------ | --------------------------- |
| `opencode` | normalized JSON export (`--conversations`) | `@opencode-ai/plugin` hooks |

Add an agent by implementing `HarnessAdapter` (`src/core/types.ts`) in one file under `src/harness/`
and registering it in `src/harness/registry.ts`. Select it with `--harness` (backfill) or the
`harness` config key (runtime, default `opencode`).

## Backfill

Shared settings (`--bank`, `--api-url`, `--api-token`, `--harness`) default from the config file
(below); the flags override per run. Operation flags are CLI-only.

```bash
hindsight-coding-backfill \
  --repo /path/to/repo \
  [--bank myproject] [--harness opencode] \
  [--conversations sessions.json] [--api-url http://localhost:8888] [--api-token X] \
  [--config <path>] [--limit 100] [--reset] [--no-pages] [--concurrency 8]
```

`sessions.json` (opencode export): `[{ "id": "s1", "turns": [{"role":"user","text":"..."}, {"role":"assistant","text":"..."}] }, ...]`

Tip: run with `--limit 100` first to validate the setup before a full-history ingest.

## Configuration

All configuration is a single JSON file — **`~/.hindsight/coding-agent.json`** — read by both the
runtime plugin and the backfill CLI (there are **no environment variables**). The file is optional;
every field has a default.

```jsonc
{
  "apiUrl": "http://localhost:8888", // Hindsight API base URL
  "apiToken": "...", // bearer token (optional)
  "bankId": "myproject", // memory bank id (default "coding")
  "harness": "opencode", // runtime adapter (default "opencode")
  "disabled": false, // hard off-switch — inert plugin, for a no-memory baseline
  "retainSessions": false, // enable live write-back
  "retainEveryTurns": 5, // write-back cadence (user turns)
  "reflectTimeoutMs": 120000, // reflect timeout
  "gitSync": {
    // incremental on-load git-sync
    "enabled": false, //   off by default; set true to keep the bank current with new commits
    "ref": "origin/main", //   sync target ref (falls back to HEAD if absent)
    "fetch": false, //   git fetch the ref before diffing (no network by default)
  },
}
```

## Plugin (opencode)

Register the plugin in `opencode.json` (all behaviour is configured via the JSON file above):

```json
{ "plugin": ["/path/to/hindsight-coding-agents"] }
```

The agent also gets a **`memory_reflect`** tool automatically (opencode) — no config needed.

Local Hindsight: `docker run -d -p 8888:8888 -p 9999:9999 -e HINDSIGHT_API_LLM_PROVIDER=gemini -e HINDSIGHT_API_LLM_API_KEY=$GEMINI_API_KEY -e HINDSIGHT_API_LLM_MODEL=gemini-2.5-flash ghcr.io/vectorize-io/hindsight:latest`

## Layout

```
src/
  core/        # harness-agnostic: config, hindsight client, missions, git + chat ingest, git-sync, inject, RuntimeCore
  harness/     # per-agent adapters + registry (opencode)
  index.ts     # opencode runtime entrypoint (loads config -> resolves harness -> adapter -> RuntimeCore)
  backfill.ts  # CLI (config + --flags; resolves --harness -> adapter.chatReader; shared git/chat/pages ingest)
```
