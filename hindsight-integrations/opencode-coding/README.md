# hindsight-coding-opencode

Reflect-only [Hindsight](https://vectorize.io/hindsight) long-term memory for coding agents in
[OpenCode](https://opencode.ai), plus a one-shot **backfill** that ingests a repo's git history and
past developer conversations into a Hindsight bank.

## What it does

- **Runtime plugin** — exposes a single tool, `hindsight_reflect(question)`. The agent asks memory a
  question phrased from the bug's symptom; Hindsight `reflect` returns a synthesized, root-cause
  answer drawn from the ingested history. No recall, no auto-injection.
- **Backfill command** — `hindsight-coding-backfill`:
  1. creates/updates the bank with the reflect mission, **observations disabled**, and two named
     **retain strategies** — `git` and `chat`;
  2. ingests **every** git commit (full message + full diff, no pre-filtering) under `git`;
  3. ingests each developer conversation **raw** (never pre-summarized) under `chat`;
  4. tags every item with a `REF-ID` so a reflected fact traces back to its commit/session.

Git and chat use **different Hindsight retain strategies** (per-item `strategy`), so each content
type is extracted with settings suited to it — in one bank, one pass.

## Backfill

```bash
hindsight-coding-backfill \
  --repo /path/to/repo \
  --conversations sessions.json \
  --bank myproject \
  --api-url http://localhost:8888 \
  [--limit 10] [--reset] [--concurrency 8]
```

`sessions.json`: `[{ "id": "s1", "turns": [{"role":"user","text":"..."}, {"role":"assistant","text":"..."}] }, ...]`

Tip: run with `--limit 10` first to validate the setup before a full-history ingest.

## Plugin

Add to `opencode.json` and configure via env:

```json
{ "plugin": ["/path/to/hindsight-coding-opencode"] }
```

```bash
HINDSIGHT_API_URL=http://localhost:8888   # default
HINDSIGHT_BANK_ID=myproject
# HINDSIGHT_API_TOKEN=...    (optional)
# HINDSIGHT_DISABLED=1       (hard off-switch — inert plugin, for a no-memory baseline)
```

Local Hindsight: `docker run -d -p 8888:8888 -p 9999:9999 -e HINDSIGHT_API_LLM_PROVIDER=gemini -e HINDSIGHT_API_LLM_API_KEY=$GEMINI_API_KEY -e HINDSIGHT_API_LLM_MODEL=gemini-2.5-flash ghcr.io/vectorize-io/hindsight:latest`
