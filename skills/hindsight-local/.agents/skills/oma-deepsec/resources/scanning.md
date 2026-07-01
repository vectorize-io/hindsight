# Scanning: `scan` → `process` → `triage` → `revalidate` → `export`

All commands run from inside `.deepsec/`. `bunx deepsec …` is interchangeable with `pnpm deepsec …`, `npm exec deepsec …`, `yarn deepsec …`.

## Pipeline

```
scan          process        revalidate          enrich           export / report / metrics
  │             │                │                │                  │
  ▼             ▼                ▼                ▼                  ▼
candidates → findings   TP/FP/Fixed verdict  → +committers      JSON / md-dir / aggregate
                                                +ownership
```

Stages are idempotent and additive. Re-running merges new info instead of overwriting. State lives under `data/<id>/`.

## Calibration first (mandatory on > 500-file repos)

The deepsec docs (`getting-started.md`, `vercel-setup.md`, `faq.md`) recommend `--limit 50 --concurrency 5` as the calibration starting point. Defer to a user-named value if given.

```bash
bunx deepsec scan
bunx deepsec status                                   # show pending / scanned counts
bunx deepsec process --limit 50 --concurrency 5       # upstream-recommended calibration
```

`scan` runs ~110 regex matchers across the codebase. **No AI calls.** ~15s on 2k files. Output goes to `data/<id>/files/` as one `FileRecord` JSON per scanned source file.

The calibration `process` is a budget-capped AI pass. Read the per-batch cost the CLI prints, multiply by `(total_files / 50)` to extrapolate. **Get the user's explicit go-ahead before launching the unbounded `process`.**

## Cost guide (`--agent claude`, Claude Opus — the most expensive backend)

| Files | Approx cost | Approx wall time |
|---|---|---|
| 100 | $25–60 | 5–15 min |
| 500 | $130–300 | 25–60 min |
| 2,000 | $500–1,200 | 1.5–4 hr |

Costs swing 2–3× based on file complexity. Codex is cheaper per call; Opus is the precision benchmark.

## Full investigation

```bash
bunx deepsec process --concurrency 5
```

Defaults: `--agent codex` (`gpt-5.5`) unless `defaultAgent` in `deepsec.config.ts` pins otherwise; `--agent claude` uses `claude-opus-4-8`. `--batch-size 5`; `--concurrency` defaults to cores−1 (the cost guide above assumes `--concurrency 5` ⇒ 25 files in flight at peak). Files are claimed atomically via `lockedByRunId`; multiple workers can run in parallel without stepping on each other.

For the precision (most expensive) backend:

```bash
bunx deepsec process --agent claude
```

Codex (the default) runs in a strict read-only sandbox and is fast at grep-heavy investigations. Backends mix freely within a project: re-process unconvincing findings with the other agent, and findings dedupe across agents.

### Resume after interruption

`process` and `revalidate` are safe to re-run. Network blip, transient model error, quota stop, Ctrl-C → re-run the **same** command. Files already finished are skipped. **Nothing to clean up.** Never `rm -rf data/<id>/` to "start clean" without explicit user instruction.

### Reinvestigate finished work

Use `--reinvestigate` (entire repo) or `--reinvestigate <N>` (wave marker) when a stronger model lands or you want a second opinion. Findings dedupe across agents; the new analysis appends to `analysisHistory` rather than overwriting.

## Triage and revalidate

```bash
bunx deepsec triage --severity HIGH
bunx deepsec revalidate --min-severity HIGH
```

| Stage | What | Cost |
|---|---|---|
| `triage` | Classifies findings P0/P1/P2/skip from finding text only (no code re-read). Claude Sonnet by default. | ~$0.01 / finding |
| `revalidate` | Re-reads code + git history, emits `true-positive` / `false-positive` / `fixed` / `uncertain` verdicts and may adjust severity. | Comparable to `process` |

`revalidate` empirically cuts FP rate by 50%+ on most repos. Run it on `HIGH+` before surfacing anything to the user.

## Export

```bash
bunx deepsec export --format md-dir --out ./findings        # one .md per finding under {CRITICAL,HIGH,…}/
bunx deepsec export --format json   --out findings.json     # single JSON array, pipe-friendly
bunx deepsec metrics                                        # aggregate counts, severities, TP rates
bunx deepsec report                                         # per-project markdown + JSON summary
```

Each command takes `--project-id <id>` if your config has multiple projects.

## Useful flags

| Flag | Purpose |
|---|---|
| `--limit <N>` | Cap files processed in this run. |
| `--concurrency <N>` | Parallel batches in flight. Lower for laptop-friendliness or quota-friendliness. |
| `--batch-size <N>` | Files per batch (default 5). |
| `--max-turns <N>` | Cap agent conversation turns per batch. |
| `--agent claude|codex` | Backend selection. |
| `--model <id>` | Override per-backend model (`claude-sonnet-4-6`, `gpt-5.5-pro`, `claude-haiku-4-5`, …). |
| `--matchers <slugs>` | CSV of slugs; restricts the matcher set on `scan`. Overrides `matchers.only` in config when both are set. |
| `--reinvestigate` / `--reinvestigate <N>` | Force re-analysis on `process`. |
| `--force` | Force re-analysis on `revalidate`. |
| `--project-id <id>` | Pick a project when more than one is registered. |
| `--root <path>` | Override project root for one-off scans. |

## Reading `data/` directly

`data/<id>/files/**/*.json` are `FileRecord`s. Useful jq one-liners:

```bash
# All TP HIGH+ findings
jq -r '. as $r | $r.findings[] | select(.revalidation.verdict=="true-positive") | select(.severity=="HIGH" or .severity=="CRITICAL") | [$r.filePath, .severity, .title] | @tsv' data/<id>/files/**/*.json

# Total spend on this project
jq -s 'map(.analysisHistory[].costUsd // 0) | add' data/<id>/files/**/*.json

# Files still pending after the latest run
jq -r 'select(.status=="pending") | .filePath' data/<id>/files/**/*.json
```

For richer queries, prefer `bunx deepsec export --format json`. Its filters match the rest of the CLI.

## Cron / scheduled CI

```bash
# Sunday cron: full scan
bunx deepsec scan
bunx deepsec process --concurrency 5
bunx deepsec revalidate --min-severity HIGH
bunx deepsec export --format json --out findings.json
```

Persist `.deepsec/data/` between runs (cache it as a build artifact) or re-scan from scratch each time. The append-only model means cached `data/` strictly improves cost on the next run.

## Distributed (`sandbox`)

Large monorepos can fan work across Vercel Sandbox microVMs:

```bash
bunx deepsec sandbox process --project-id my-app --sandboxes 10 --concurrency 4
```

Local working tree is tarballed (`.git` excluded) and uploaded. Sandbox-level network egress is locked to the configured AI host(s); the gateway key is injected outside the sandbox so it cannot be exfiltrated. Use this when the repo is large enough that local concurrency saturates your machine, or when running unattended in CI/CD.

See `config.md` for Sandbox auth (OIDC vs access token).

## What `process` does not do

- Does not modify source code. Findings are advisory.
- Does not commit / push / open PRs. Hand off to `oma-scm` if the user wants commits.
- Does not call out to non-AI external services unless a notifier plugin is configured.
- Does not phone home or report telemetry; `data/<id>/` stays on your machine unless explicitly exported.
