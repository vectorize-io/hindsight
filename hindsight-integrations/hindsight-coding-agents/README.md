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

| harness | past-session source (backfill) | runtime binding |
|---|---|---|
| `opencode` | normalized JSON export (`--conversations`) | `@opencode-ai/plugin` hooks |

Add an agent by implementing `HarnessAdapter` (`src/core/types.ts`) in one file under `src/harness/`
and registering it in `src/harness/registry.ts`. Select it with `--harness` (backfill) or
`HINDSIGHT_HARNESS` (runtime, default `opencode`).

## Backfill

```bash
hindsight-coding-backfill \
  --repo /path/to/repo \
  --bank myproject \
  --harness opencode \
  --conversations sessions.json \
  --api-url http://localhost:8888 \
  [--limit 100] [--reset] [--no-pages] [--concurrency 8]
```

`sessions.json` (opencode export): `[{ "id": "s1", "turns": [{"role":"user","text":"..."}, {"role":"assistant","text":"..."}] }, ...]`

Tip: run with `--limit 100` first to validate the setup before a full-history ingest.

## Plugin (opencode)

Add to `opencode.json` and configure via env:

```json
{ "plugin": ["/path/to/hindsight-coding-agents"] }
```

```bash
HINDSIGHT_API_URL=http://localhost:8888   # default
HINDSIGHT_BANK_ID=myproject
HINDSIGHT_HARNESS=opencode                # default; selects the runtime adapter
# HINDSIGHT_API_TOKEN=...           (optional)
# HINDSIGHT_RETAIN_SESSIONS=1       (enable live write-back)
# HINDSIGHT_RETAIN_EVERY_TURNS=5    (write-back cadence)
# HINDSIGHT_DISABLED=1              (hard off-switch — inert plugin, for a no-memory baseline)
```

Local Hindsight: `docker run -d -p 8888:8888 -p 9999:9999 -e HINDSIGHT_API_LLM_PROVIDER=gemini -e HINDSIGHT_API_LLM_API_KEY=$GEMINI_API_KEY -e HINDSIGHT_API_LLM_MODEL=gemini-2.5-flash ghcr.io/vectorize-io/hindsight:latest`

## Layout

```
src/
  core/        # harness-agnostic: hindsight client, missions, git + chat ingest, inject, RuntimeCore
  harness/     # per-agent adapters + registry (opencode)
  index.ts     # opencode runtime entrypoint (resolves HINDSIGHT_HARNESS -> adapter -> RuntimeCore)
  backfill.ts  # CLI (resolves --harness -> adapter.chatReader; shared git/chat/pages ingest)
```
