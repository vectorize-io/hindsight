---
name: docs
description: Documentation drift detection and sync via `oma-docs`. Verify mode finds broken refs in all repo markdown (default glob `**/*.md`), sync mode proposes patches for docs affected by a git diff.
disable-model-invocation: true
---

# MANDATORY RULES: VIOLATION IS FORBIDDEN

- **Response language follows `language` setting in `.agents/oma-config.yaml` if configured.**
- **NEVER skip steps.** Execute from Step 1 in order.
- **Never auto-apply sync patches.** Sync mode is always interactive: `[y]` confirm required per doc.
- **Never modify `.agents/`.** SSOT protection applies in all modes.
- **Follow the host-LLM contract** in `.agents/skills/oma-docs/SKILL.md`: the CLI emits structured data; this workflow performs natural-language synthesis, severity grouping, and patch drafting on top of the JSON output.

---

> **Vendor note:** This workflow executes inline (no subagent spawning). All vendors invoke `oma docs` directly.

---

## L1 Decision Events

Emit required L1 decisions by calling `oma state:emit` directly, as documented in `.agents/skills/_shared/runtime/event-spec.md`.

---

## Step 1: Detect Mode

Inspect the user's request to select a mode:

| Mode | Triggers |
|------|----------|
| `sync` | Prompt mentions `sync`, "동기화", "patch docs", "update docs after change", or supplies a git diff range (e.g. `HEAD~1..HEAD`, `main..feature`). |
| `verify` | Default. Use when the request is about checking, auditing, or validating docs. |

If intent is ambiguous, ask once:

```
Run `oma docs verify` (drift check) or `oma docs sync` (propose patches for a git diff)?
```

Capture optional arguments from the prompt:
- **verify**: glob path (e.g. `docs/**/*.md`, `cli/README.md`), `--no-urls`, `--urls-sync`, `--report-file <path>`.
- **sync**: git diff range (default: staged, fallback `HEAD~1..HEAD`).

---

## Step 2: Preflight

1. Confirm `oma` is available: `command -v oma` (or `bun run oma --help` if running from source).
2. For `sync` mode, confirm the repo has a usable diff:
   - If `--cached` returns nothing, fall back to `HEAD~1..HEAD`.
   - If neither is available, ask the user for an explicit range.
3. If `oma docs` is missing entirely, print an install hint and exit. Do NOT silently substitute manual greps.

---

## Step 3A: Verify Mode

// turbo
Run the deterministic drift check and capture JSON for downstream synthesis:

```bash
oma docs verify --json
```

Variants (apply only the flags the user requested):

```bash
# Narrow scope
oma docs verify "docs/**/*.md" --json
oma docs verify cli/README.md --json

# Persist a full markdown report
oma docs verify --report-file ./drift-report.md

# Skip URL checking (if lychee is unavailable or run separately)
oma docs verify --no-urls --json

# Block until lychee URL check finishes (CI-style)
oma docs verify --urls-sync --json
```

Exit codes:
- `0`: clean.
- `1`: broken refs found in core check (URL drift does NOT affect this exit code; see `docs/generated/url-drift.json`).

---

## Step 3B: Sync Mode

Run candidate-doc lookup against the user-supplied range:

```bash
# Default: staged changes; fallback HEAD~1..HEAD
oma docs sync --json

# Explicit range
oma docs sync HEAD~5..HEAD --json
oma docs sync main..feature-branch --json
```

The CLI emits a list of `{ doc, changedFiles, matchedRefs }` entries. **Do not auto-apply anything.** Patch synthesis is your responsibility (host-LLM contract).

---

## Step 4: Synthesize Findings (Host-LLM Contract)

### Verify mode

Read the JSON drift report and:

1. Group findings by severity / kind:
   - **CRITICAL**: broken `file` refs in critical paths (CLAUDE.md, top-level READMEs, install docs).
   - **HIGH**: broken `cli`, `script`, `env`, `config` refs anywhere in `docs/`.
   - **MEDIUM**: broken `file` refs in deeper documentation sections.
   - **LOW**: URL drift surfaced in `docs/generated/url-drift.json` (when present).
2. For each finding, suggest a concrete fix (renamed path, missing CLI install, removed env var, etc.).
3. Prioritize fixes for files most central to the project.
4. If the user asks for a natural-language summary, generate it from the JSON, never from cached prose.

### Sync mode

For each candidate doc:

1. Read the doc itself.
2. Read `git diff` for the listed `changedFiles`.
3. Draft a unified-diff patch reflecting the code change. Keep the patch minimal: only update text that the diff actually invalidates.
4. Present each patch to the user with the prompt template:

   ```
   [y] apply  [n] skip  [d] show diff  [s] show full proposal
   ```

5. After each `[y]` or `[n]` decision, emit and verify the required patch approval decision:

   ```bash
   oma state:emit "decision.made" '{"subject":"docs.sync-patch-approval","decision":"Apply or skip the proposed documentation sync patch for this document.","rationale":"The user reviewed the proposed doc patch and made an explicit per-document decision."}'
   oma state:verify --workflow docs --checkpoint sync-patch-approval
   ```

6. On `[y]`, apply via `git apply` or by writing the doc directly. After applying any patches, regenerate the index:

   ```bash
   oma docs verify --json > /dev/null
   ```

   (verify always overwrites `docs/generated/doc-refs.json`.)

---

## Step 5: Report

Tell the user:

- Mode executed (`verify` / `sync`).
- Counts: broken refs by kind (verify), candidate docs / applied patches (sync).
- Top 3 actionable items with `file:line` references.
- Pointer to `docs/generated/doc-refs.json` and (if applicable) `docs/generated/url-drift.json`.
- Any skipped checks (e.g. `lychee` missing, LLM unavailable, secret-bearing files excluded).

**Verify report template:**

```markdown
## Docs Verify Report
- Scope: **/*.md repo-wide, or the requested glob (N docs scanned)
- Broken: file=A cli=B script=C env=D config=E
- Top fixes:
  1. <file:line> — <description> → <fix>
  2. ...
- URL drift: see docs/generated/url-drift.json (M flagged)
```

**Sync report template:**

```markdown
## Docs Sync Report
- Range: <range>
- Candidate docs: N
- Applied patches: M (user-confirmed)
- Skipped: K (user declined or no actionable change)
- Index regenerated: docs/generated/doc-refs.json
```

---

## Failure Handling

| Situation | Recovery |
|-----------|----------|
| `oma` not on PATH | Print install hint; exit. Do not fall back to manual grep. |
| `lychee` missing | Print install hint (`brew install lychee`); continue with core check only. |
| `doc-refs.json` stale in sync | Run `oma docs verify --json` first, then re-run sync. |
| LLM unavailable for verify summary | Emit raw JSON drift report and let the user review. |
| LLM unavailable for sync proposals | Emit candidate-list-only output; user reviews matched refs manually. |
| Extractor parse error on a single doc | Skip + warn; continue with remaining docs. |
| `git apply` fails on an approved patch | Show the failure; offer to write the doc directly or skip. |

---

## Quick Reference

| Command | Effect |
|---------|--------|
| `/docs` | Verify all docs (default mode). |
| `/docs verify "docs/**/*.md"` | Verify a glob scope. |
| `/docs verify --report-file ./drift.md` | Persist full markdown report. |
| `/docs sync` | Propose patches for staged changes. |
| `/docs sync HEAD~5..HEAD` | Propose patches for a commit range. |
| `/docs sync main..feature` | Propose patches for a branch diff. |

---

## References

- Skill spec: `.agents/skills/oma-docs/SKILL.md`
- Design doc: `docs/plans/designs/008-oma-docs.md`
- CLI source: `cli/commands/docs/`
- Workflow hook (auto verify on `/scm`, `/work`, `/ultrawork`): toggle via `docs.auto_verify` in `.agents/oma-config.yaml`.
