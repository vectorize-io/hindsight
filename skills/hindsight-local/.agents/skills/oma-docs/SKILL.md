---
name: oma-docs
description: Verify documentation references against the current codebase and propose updates for diff-affected docs. Use to check if docs still match reality (broken file paths, CLI commands, config keys, env vars, scripts) and to surface docs that may need updating after code changes.
---

# oma-docs - Documentation Drift Detector

## Scheduling

### Goal
Detect broken references in `docs/**/*.md` (verify mode) and propose LLM-generated patch proposals for docs affected by recent code changes (sync mode). Both modes run on-demand; sync is always interactive.

### Intent signature
- User asks to check if docs are up to date, find broken doc links, verify file paths referenced in docs, or detect documentation drift.
- User asks to update docs after a code change, propose doc patches for a git diff, or sync affected docs.
- A workflow hook checks `docs.auto_verify: true` and runs `oma docs verify --json` at completion.

### When to use
- After a refactor, rename, or file deletion, to find stale references in docs.
- Before a release, to confirm that CLI commands, file paths, and config keys in docs still exist.
- After a significant git diff, to discover which docs reference the changed files and may need updating.
- Routine drift check on any docs-heavy repo.

### When NOT to use
- Generating docs from scratch for undocumented features → v2 create mode.
- Multilingual translation of docs → use `oma-translator`.
- Symbol-level semantic drift (function signature changes not reflected in prose) → v2 L3 mode.
- CI-blocking enforcement → v2 block mode (v1 is warn-only).

### Expected inputs

**verify mode**: Optional glob path (default `**/*.md`), optional `--json` flag, optional `--report-file <path>`.

**sync mode**: Optional git diff range (default `--cached`, fallback `HEAD~1..HEAD`).

### Expected outputs

**verify mode**:
- Markdown drift report to stdout (default), or raw JSON with `--json`, or full markdown written to file with `--report-file`.
- Exit code 0 if clean, 1 if any broken refs found.

**sync mode**:
- Per-doc patch proposals drafted by the host LLM from the CLI's candidate-doc list, confirmed per doc (`[y] apply [n] skip [d] diff [s] full proposal` style).
- Docs modified only on explicit user approval; `doc-refs.json` regenerated after applies.

### Dependencies
- `cli/commands/docs/extract.ts`: markdown AST + L2 pattern extractor.
- `cli/commands/docs/resolve.ts`: deterministic broken-ref checker.
- `cli/commands/docs/reporter.ts`: deterministic markdown/JSON report renderer (no LLM call; host LLM does narrative synthesis).
- `cli/commands/docs/sync-propose.ts`: git diff intake, reverse lookup, candidate-doc selector with secret redaction (no LLM call; host LLM drafts patches).
- `docs/generated/doc-refs.json`: single-direction reference index (git-tracked, regenerated on every verify run).
- `docs/generated/url-drift.json`: lychee-produced URL drift report (written by background lychee spawn; gitignored or tracked at user discretion).
- `lychee`: external Rust tool for URL link checking. Detected on PATH; install via `brew install lychee` or see https://github.com/lycheeverse/lychee#installation. Optional but recommended.
- `.agents/oma-config.yaml`: `docs.auto_verify` (workflow hook opt-in), `docs.check_urls` (URL checking on/off, default true), and `docs.exclude` (glob list of markdown trees the walker must not scan — benchmark artifacts, translation mirrors, etc.; default `[]`) toggles.

### Control-flow features
- Mode is selected from the first argument: `verify` or `sync`.
- verify: extract → resolve → report (fully deterministic CLI; host LLM adds narrative summary on top of the JSON/markdown output).
- sync: git diff → reverse lookup → candidate list (CLI) → host-LLM patch proposals → interactive accept/reject.
- Branches on `--json`, `--report-file`, LLM availability, and network reachability.
- Never blocks workflow completion in v1 (warn-only hook policy).

## Structural Flow

### Entry
1. Read first argument to select mode (`verify` | `sync`). If absent, print help and exit.
2. Load `oma-config.yaml` to check `docs.auto_verify` when invoked from a workflow hook.
3. Confirm required CLI dependencies (`oma docs verify`, `oma docs sync`) are on PATH.

### Scenes
1. **PREPARE**: Determine mode, resolve path/diff-range arguments, confirm tool availability.
2. **ACQUIRE**: Run extractor (`extract.ts`) to regenerate `doc-refs.json` from `docs/**/*.md` (verify) or build in-memory reverse index from existing `doc-refs.json` (sync).
3. **REASON**: Resolve each reference deterministically (verify) or correlate changed files to candidate docs via reverse lookup (sync).
4. **ACT**: Render the deterministic drift report (verify) or list candidate docs with matched refs (sync). Host LLM does any natural-language synthesis or patch drafting on top of this output.
5. **VERIFY**: Confirm output shape is valid (JSON schema check for `--json`; structured candidate list for sync).
6. **FINALIZE**: Print to stdout, write report file if requested, emit exit code.

### Transitions
- verify mode: PREPARE → ACQUIRE (extract) → REASON (resolve) → ACT (report) → FINALIZE.
- sync mode: PREPARE → ACQUIRE (reverse index) → REASON (candidate matching) → ACT (LLM proposals) → VERIFY (interactive) → FINALIZE (apply approved).
- If LLM is unavailable in verify: skip reporter summary, emit raw JSON drift report.
- If LLM is unavailable in sync: emit candidate-list-only output (no patch proposals); user reviews manually.
- If `doc-refs.json` is stale or missing in sync: run extractor first, then continue.

### Failure and recovery
- Extractor parse error on a single doc: skip doc + warn, continue with remaining docs.
- lychee unavailable or URL check incomplete: print install hint, skip URL checking, continue (core check is unaffected).
- Host-LLM context limit exceeded while drafting patches: process candidate docs in smaller batches.
- `oma docs` CLI not found: skip with installation hint (workflow hook: skip silently).
- `doc-refs.json` write failure: abort and report the write error; do not emit partial index.

### Exit
- Success (verify): drift report emitted; exit 0 if clean, exit 1 if broken refs found.
- Success (sync): approved patches applied; `doc-refs.json` regenerated; session summary printed.
- Partial success: extractor or resolver errors are explicit in the report; no silent failures.

## Logical Operations

### Actions
| Action | SSL primitive | Notes |
|--------|---------------|-------|
| Parse CLI args and mode | `READ` | First arg selects verify or sync |
| Extract refs from docs | `CALL_TOOL` | `extract.ts`: remark AST + L2 patterns → `doc-refs.json` |
| Check broken refs | `RESOLVE` | `resolve.ts`: file, url, cli, script, env, config checks |
| Build reverse index | `INFER` | `sync-propose.ts`: in-memory map from `doc-refs.json` |
| Match diff to candidate docs | `RESOLVE` | `sync-propose.ts`: git diff + reverse lookup |
| Redact secrets from diff | `VALIDATE` | Exclude `.env*`, `*.pem`, `*.key`, `id_rsa*`; sanitize content |
| Generate patch proposals | `INFER` | Host LLM drafts patches from `sync-propose.ts` candidate output (no CLI LLM call) |
| Render drift report | `RENDER` | `reporter.ts`: markdown (default), JSON (`--json`), file (`--report-file`) |
| Apply approved patches | `WRITE` | `git apply` on user-confirmed patches only |
| Notify hook summary | `NOTIFY` | 1-3 line stdout summary for workflow hooks |

### Tools and instruments
- `cli/commands/docs/extract.ts`: `remark` + `unified` markdown AST, L2 pattern extraction, escape hatch filter, `docs/generated/doc-refs.json` writer.
- `cli/commands/docs/resolve.ts`: case-sensitive file existence, `which` for CLI tokens, `package.json` scripts lookup, ripgrep/git grep for env vars, `oma-config.yaml` deep-path check. Per-target dedupe caches (cli by first token, env, config) and per-directory listing cache for file resolution. URL kind is filtered out by the verify command and delegated to lychee.
- `cli/commands/docs/reporter.ts`: deterministic markdown + JSON renderer. **No LLM call.** Friendly summary, severity tagging, fix prioritization are the host LLM's responsibility.
- `cli/commands/docs/sync-propose.ts`: git diff intake, reverse index build, secret-pattern + gitignore file exclusion. Returns candidate docs with matched refs only. **No LLM call.** Patch synthesis is the host LLM's responsibility.
- External: [`lychee`](https://github.com/lycheeverse/lychee) (background URL link checking; install via `brew install lychee`).

### Host-LLM contract

This skill follows the OMA pattern (mirroring `oma-scholar`): **the CLI emits structured data; the host LLM (the agent runtime that invoked the skill) does any natural-language synthesis or judgment.**

After `oma docs verify --json`:
1. Read the JSON drift report.
2. Group findings by severity / urgency (host-LLM judgment).
3. Suggest fixes per finding, prioritizing files most central to the project.
4. If the user asks for natural-language summary, host LLM produces it from the JSON, never from cached prose.

After `oma docs sync <range> --json`:
1. Read the candidate doc list (each entry: `{ doc, changedFiles, matchedRefs }`).
2. For each candidate doc: read the doc itself, read `git diff` for `changedFiles`, draft a unified-diff patch reflecting the code change.
3. Present patches to the user for review. **Never auto-apply.**
4. On user approval, apply via `git apply` or by writing the doc directly.

### Canonical command path

**verify mode** runs a drift check against the current codebase:

```bash
# Default: scan all docs/**/*.md, render markdown to stdout.
# URL link checking is delegated to lychee in the background
# (install: `brew install lychee`). Core check ~8s on a 1k-doc repo.
oma docs verify

# Narrow to a path or glob (uses minimatch)
oma docs verify "docs/**/*.md"
oma docs verify cli/README.md

# Machine-readable output for CI / hooks
oma docs verify --json

# Persist full markdown report to a file
oma docs verify --report-file ./drift-report.md

# Skip URL checking entirely (when lychee is run separately, or as a
# one-off override of docs.check_urls=true in oma-config.yaml)
oma docs verify --no-urls

# Block until lychee finishes (CI scenarios needing complete URL data)
oma docs verify --urls-sync

# Exit code: 0 = clean, 1 = broken refs found in core check.
# URL drift, if any, is reported separately at docs/generated/url-drift.json
# and does NOT affect this exit code.
```

**sync mode** proposes patches for docs affected by a git diff (always interactive, never auto-applies):

```bash
# Default: staged changes (--cached), fallback HEAD~1..HEAD
oma docs sync

# Explicit range
oma docs sync HEAD~5..HEAD
oma docs sync main..feature-branch

# The CLI emits the candidate-doc list; the host LLM drafts patches and
# confirms per doc ([y] apply / [n] skip / [d] diff / [s] full proposal).
# Sync regenerates docs/generated/doc-refs.json after applying any patches.
```

**Workflow hook (opt-in)** runs verify automatically at workflow completion when `docs.auto_verify: true` in `oma-config.yaml`:

```bash
# Hook command emitted by /scm, /work, /ultrawork
oma docs verify --json
# Hook policy: warn-only in v1; non-zero exit does NOT block workflow completion
```

### Resource scope
| Scope | Resource target |
|-------|-----------------|
| `LOCAL_FS` read | `docs/**/*.md` (extractor input), `docs/generated/doc-refs.json` (index), `.env.example`, `package.json`, `.agents/oma-config.yaml` |
| `LOCAL_FS` write | `docs/generated/doc-refs.json` (regenerated each verify run), approved sync patches |
| `CODEBASE` read-only | Existence checks for file/cli/script/env/config refs; git diff intake |
| `PROCESS` | `git diff`, `git apply`, `which`, background `lychee` spawn |
| `NETWORK` | URL checking delegated to `lychee` (no internal HEAD fallback; see Guardrail 6) |

### Preconditions
- `docs/` directory exists at repo root.
- `cli/commands/docs/` is built and `oma` binary is on PATH (or invoked directly via `bun run`).
- For sync mode: a git diff is available (`--cached` stage or recent commits).

### Effects and side effects
- verify: regenerates `docs/generated/doc-refs.json` (always overwrites).
- sync: modifies docs files only on user approval; regenerates `doc-refs.json` after applies.
- Both modes: stdout output (summary or full report).
- No `.agents/` files are ever modified.

### Guardrails

1. **Never modify `.agents/`**: CLAUDE.md SSOT protection applies in all modes.
2. **Never auto-apply sync patches**: sync is always interactive; `[y]` confirm required per doc.
3. **LLM unavailable → graceful degradation**: verify falls back to raw JSON; sync falls back to candidate-list-only (no proposals). Neither mode blocks on LLM availability.
4. **Response language follows `oma-config.yaml` `language`**: user-facing report text is localized; code, paths, JSON keys, and CLI commands stay in English.
5. **Secret-bearing files excluded from sync output**: `.env*`, `*.pem`, `*.key`, `id_rsa*`, and gitignored files never appear in candidate `changedFiles` lists. Host LLM never sees secret file paths.
6. **URL link checking delegated to lychee**: when `docs.check_urls=true` (default), URL refs are checked by `lychee` running in the background; results land in `docs/generated/url-drift.json`. If `lychee` is missing, an install hint is printed and URL checking is skipped (no internal HEAD fallback).
7. **No direct LLM API calls from the CLI**: the CLI never imports vendor SDKs, never reads API keys, never makes outbound LLM requests. All synthesis, patch drafting, and natural-language framing is the host LLM's responsibility (mirrors `oma-scholar`'s pattern). This makes `oma-docs` vendor-agnostic: works identically under Claude Code / Codex / Gemini / Qwen / Antigravity.
8. **Hook is warn-only in v1**: broken refs never block workflow completion; `docs.auto_verify: false` by default (explicit opt-in required).
9. **Escape hatch respected**: `<!-- oma-docs:ignore-start -->` / `<!-- oma-docs:ignore-end -->` blocks and frontmatter `oma-docs: skip` are honored; no ref extraction from ignored regions. Use this for illustrative example paths in tutorials (hypothetical project files in inline code) that intentionally do not resolve.
10. **Gitignored targets are generated, not broken**: a `file` ref whose target does not exist but matches the project's gitignore rules (`git check-ignore`) is classified as `skipped` (a documented runtime/generated output such as `.agents/results/result-*.md`, `.serena/memories/*`, `.agents/state/*.json`), never as `broken`. gitignore is the single source of truth for "produced at runtime" — gitignore an output path and it stops being flagged. The `skipped` count is surfaced (markdown summary + JSON `skippedCount`) so nothing is silently dropped.
11. **Non-prose trees excluded via `docs.exclude`**: committed-but-non-prose markdown (benchmark run artifacts, translation mirrors validated separately by `oma docs i18n`) is dropped from the scan by the `docs.exclude` globs rather than producing unactionable broken refs. An explicit single-file path argument bypasses `docs.exclude`.

### v1 scope note
v1 covers `verify` and `sync` (broken-only classification, L2 ref extraction). The following are explicitly deferred to v2: `create` mode (generate missing docs), multilingual sync (deeper `oma-translator` integration), L3 symbol-level extraction (Tree-sitter/LSP), GitHub Action wrapper, `block` hook mode.

## References
- Design doc: `docs/plans/designs/008-oma-docs.md` (full architecture, schema spec, decision log, edge cases).
- Schema spec: `doc-refs.json` v1 schema defined in design doc § doc-refs.json Schema.
- Workflow hook integration: design doc § Workflow Hook Integration.
- Migration: `deepinit` Step 6 retirement, design doc § Migration: deepinit Step 6.
- Adjacent skills: `oma-translator` (v2 multilingual), `oma-skill-creator` (SSL-lite validation).
