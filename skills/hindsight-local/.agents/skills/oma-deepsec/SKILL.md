---
name: oma-deepsec
description: >
  Drive Vercel's `deepsec` agent-powered vulnerability scanner end-to-end:
  installing the `.deepsec/` workspace, bootstrapping `INFO.md`, running
  cost-aware `scan` / `process` / `triage` / `revalidate` / `export` passes,
  gating PRs with `process --diff`, writing custom matchers, and triaging
  findings. Use whenever the user mentions deepsec, asks an agent to scan a
  repo for vulnerabilities, runs into `pnpm deepsec` / `bunx deepsec`
  commands, wants a CI-based PR security review, sees a `.deepsec/`
  directory, or asks about `INFO.md` / matchers / `process --diff` /
  `revalidate`, even when the tool name is not spoken. Deepsec scans are
  expensive (a single full scan can cost hundreds to tens of thousands of
  dollars) so the skill exists in part to keep the user from getting
  surprised.
---

# Deepsec: Agent-Powered Vulnerability Scanner Driver

## Scheduling

### Goal
Operate Vercel's `deepsec` security scanner inside a target repository safely and cost-consciously: bootstrap the `.deepsec/` workspace, write a tight `INFO.md`, run the right scan/process/triage/revalidate/export sequence, gate PRs in CI via `process --diff`, and grow project-specific matchers, surfacing real, revalidated findings without runaway spend.

### Intent signature
- User mentions `deepsec`, "deep security scan", `bunx deepsec`, `pnpm deepsec`, `npx deepsec`.
- User asks an agent to scan a repository for vulnerabilities, security issues, or CVEs and the project has (or should have) a `.deepsec/` directory.
- User asks how to add a deepsec PR / CI security gate, or about `process --diff`, `--diff-staged`, `--diff-working`, `--files-from`, `--comment-out`.
- User mentions deepsec artefacts: `INFO.md`, `SETUP.md`, `data/<id>/files/`, `FileRecord`, `RunMeta`, `revalidation`, `triage`, custom matchers, `MatcherPlugin`, `noiseTier`, `priorityPaths`.
- User asks about deepsec configuration: `deepsec.config.ts`, `defaultAgent`, `AI_GATEWAY_API_KEY`, `VERCEL_OIDC_TOKEN`, AI Gateway, Vercel Sandbox, `--agent codex`, `--agent claude`.
- User asks how to lower deepsec cost, cut false-positive rate, or interpret severity / triage / revalidation verdicts.

### When to use
- First-time deepsec install in a repo (`init`, `INFO.md` write, first calibration scan).
- Running a full or scoped scan and processing findings.
- Setting up a per-PR CI gate with `process --diff` and `--comment-out`.
- Writing a project-specific matcher to cover entry points the default set misses.
- Triaging a backlog of findings (severity bucketing, FP cuts via `revalidate`, exporting to issue tracker).
- Diagnosing deepsec failures: missing credentials, AI Gateway quota stops, refusals, sandbox auth.

### When NOT to use
- Generic OWASP / lint-style review without deepsec → use `oma-qa`.
- Generic CVE / dependency advisories → use `oma-qa` or `oma-search`.
- Architecting a brand-new SAST pipeline that is not deepsec → use `oma-architecture`.
- Writing or auditing application code itself → route to `oma-backend` / `oma-frontend` / `oma-mobile`.
- Cloud / IAM / Terraform hardening → use `oma-tf-infra` (deepsec only scans the IaC; remediation lives there).
- Pure reasoning about a finding's fix in product code → use `oma-debug` once deepsec has produced the finding.

### Expected inputs
- `target_repo_root`: absolute path of the codebase to scan (parent of `.deepsec/`).
- `intent`: one of `setup` | `scan` | `pr-review` | `matchers` | `triage` | `config` | `troubleshoot`.
- `credential_mode`: `ai-gateway-key` | `vercel-oidc` | `direct-anthropic` | `direct-openai` | `subscription`.
- `agent_choice`: `codex` (upstream default; model `gpt-5.5`) or `claude` (model `claude-opus-4-8`). Asked once before the first paid call if not already provided.
- `severity_floor`: lowest severity worth surfacing (typically `HIGH`).
- Optional: existing `.deepsec/data/<id>/`, `deepsec.config.ts`, custom matchers, CI provider.

### Expected outputs
- A working `.deepsec/` workspace registered against the target repo.
- A populated `data/<id>/INFO.md` (50-100 lines, project-specific, no line numbers).
- One or more completed `scan` → `process` (→ `triage`/`revalidate`) runs with reproducible cost notes.
- For PR mode: a CI workflow file using `process --diff <base>` with two-job split (no PR-write in PR-code job).
- For matchers: new `.deepsec/matchers/<slug>.ts` files wired through the inline plugin in `deepsec.config.ts`.
- A findings export (`md-dir` and/or `json`) plus a short summary of top severities and FP-rate notes.
- Explicit, dollar-and-time-bounded plan before any pass that may cost more than ~$25.

### Dependencies
- Node.js **22+**, plus a package manager: `bun` / `bunx` (preferred in this monorepo), `pnpm`, `npm`, or `yarn`.
- A working AI credential: `AI_GATEWAY_API_KEY=vck_…`, or `VERCEL_OIDC_TOKEN`, or direct `ANTHROPIC_AUTH_TOKEN` + `ANTHROPIC_BASE_URL`, or a logged-in `claude` / `codex` CLI subscription.
- Git (history is consulted by `revalidate` and `--diff` modes).
- Optional: Vercel Sandbox auth for `deepsec sandbox …` distributed runs.
- Reference resources under `resources/` (loaded only when the scenario requires them).

### Control-flow features
- Branches by `intent` (setup vs scan vs pr-review vs matchers vs triage vs config vs troubleshoot).
- Branches by repo size (calibrate with `--limit 50` before any large pass).
- Branches by credential source (gateway key, OIDC, direct, subscription).
- Stops on quota / credit exhaustion and resumes the same command after top-up.
- Refuses to launch an unbounded `process` when no calibration has been done and the repo is large.
- Reads codebase, writes `.deepsec/` files and CI configs, runs long-lived AI processes.

## Structural Flow

### Entry
1. Confirm whether `.deepsec/` already exists; if yes, treat the run as **incremental**, never re-init.
2. Resolve `intent` from the user prompt; if ambiguous (e.g. "scan this repo"), default to `setup` then `scan` (calibration mode).
3. Estimate scale: count source files (rough `rg --files | wc -l` excluding `node_modules`, `.git`, `dist`) to forecast cost before any AI pass.
4. Check for an AI credential in `.env.local` or shell env; if none, route to credential setup before any `process` / `revalidate` / `triage` call.
5. **Confirm agent choice with the user before the first paid call.** If `agent_choice` is not already in the prompt and `deepsec.config.ts` does not pin a `defaultAgent`, ask whether to run `codex` (`gpt-5.5`, the upstream default; runs in a strict sandbox, cheaper, grep-heavy) or `claude` (`claude-opus-4-8`; strongest reasoning, most expensive). The two backends can be mixed via `--reinvestigate` and findings dedupe across agents. Skip the question if the user has already named an agent or has explicitly delegated the decision ("just pick reasonable defaults").

### Scenes
1. **PREPARE**: Resolve intent, repo root, credential, budget cap, severity floor, agent choice. Refuse to run blind on a repo of unknown scale.
2. **ACQUIRE**: Read `.deepsec/deepsec.config.ts`, `data/<id>/project.json`, `INFO.md`, last `runs/` entries, and target-repo signals (`README`, `AGENTS.md`/`CLAUDE.md`, framework configs, route directories) needed to author or verify `INFO.md`.
3. **REASON**: Pick the smallest pass that answers the user's question. Options include `scan` only, a `--limit 50` calibration, a full `process`, `process --diff`, a matcher-authoring loop, or troubleshoot-only. Always state cost forecast and stopping condition before AI passes.
4. **ACT**: Run the planned commands from inside `.deepsec/`. For matchers, write per-slug files and wire the inline plugin. For PR mode, scaffold the two-job CI workflow.
5. **VERIFY**: Use `deepsec status`, the run's `RunMeta`, exit code (`0` clean, `1` findings produced, other = error), candidate counts, and (when present) the `--comment-out` markdown to confirm output.
6. **FINALIZE**: Summarize findings by severity and verdict, list dollar cost and wall time, name files written, and call out follow-ups (revalidate `HIGH+`, write matchers for missed entry points, persist `data/` between CI runs).

### Transitions
- If `.deepsec/` is missing and intent involves scanning → run `bunx deepsec init` (or `npx deepsec init`) and follow the printed prompt to populate `INFO.md` before any AI pass.
- If `INFO.md` is empty or template-shaped → write it (50-100 lines, project-specific, 3-5 examples per section, no line numbers, no generic CWE enumeration).
- If repo is > 500 files and no calibration has run → run a calibration pass first (deepsec docs recommend `--limit 50 --concurrency 5`) and report cost extrapolation before the full pass.
- If a `process` / `revalidate` run halts on quota → leave file locks intact, surface the exact remediation URL, **re-run the same command after top-up**.
- If the agent reports a refusal (`refused: true`) → never silently drop; document the affected files and either retry with the other backend or add the path to `config.json:ignorePaths` only if reproducible.
- If the user wants a CI gate → emit the two-job pattern (PR-code job has no `pull-requests: write`, comment job has no PR code).
- If the user wants more matcher coverage → run the matcher-authoring workflow against `data/<id>/files/` and the parent repo's entry points.

### Failure and recovery
| Failure | Recovery |
|---------|----------|
| `Missing AI credentials for --agent claude` / `codex` | Pick a credential mode (gateway key / OIDC / direct / subscription) per `resources/config.md` and write `.env.local`. |
| `401 Unauthorized` from gateway | OIDC: re-run `vercel env pull` (12 h expiry). API key: regenerate. Confirm `.env.local` is in the cwd deepsec runs from. |
| `Stopped: AI Gateway credits exhausted` | Top up via the printed URL; re-run the same command, files already done are skipped. |
| `Stopped: Claude Pro/Max subscription exhausted` | Switch to AI Gateway; subscriptions don't carry full scans. |
| Persistent refusal on a single file (>5% of batches) | Add the path to `data/<id>/config.json:ignorePaths`, or run that file alone with `--batch-size 1`. |
| FP rate too high on `HIGH+` | Run `revalidate --min-severity HIGH`; tighten `INFO.md`'s threat model and FP notes; bias matchers to `precise`. |
| `noisy` matcher wedges scanner on a 100k-file repo | Tighten `filePatterns` to language- or directory-anchored globs. |
| Sandbox auth fails | OIDC: re-run `vercel env pull`. Access-token mode: verify `VERCEL_TOKEN` + `VERCEL_TEAM_ID` + `VERCEL_PROJECT_ID`. |
| User asks for full scan with no budget context | Halt; report file count and forecast cost band; require explicit go-ahead before the full pass. |

### Exit
- **Success**: planned passes ran, findings exist with verdicts (or no findings produced), files written are listed, residual cost / followups are explicit.
- **Partial success**: some passes blocked on credentials/quota/refusal; the blocker, the safe-resume command, and the recommended next step are reported.
- **Failure**: nothing destructive happened, the user has the exact next command to unblock the work.

## Logical Operations

### Actions
| Action | SSL primitive | Evidence |
|--------|---------------|----------|
| Detect existing workspace and credentials | `READ` | `.deepsec/`, `.env.local`, env vars |
| Estimate repo scale | `INFER` | `rg --files | wc -l` |
| Choose pass plan (calibrate vs full vs diff) | `SELECT` | File count, intent, budget cap |
| Init workspace | `CALL_TOOL` | `bunx deepsec init` |
| Write `INFO.md` | `WRITE` | `data/<id>/INFO.md` |
| Run scan | `CALL_TOOL` | `bunx deepsec scan` |
| Run AI investigation | `CALL_TOOL` | `bunx deepsec process` (`--limit`, `--concurrency`) |
| Triage / revalidate | `CALL_TOOL` | `bunx deepsec triage` / `revalidate --min-severity HIGH` |
| Export findings | `CALL_TOOL` | `bunx deepsec export --format md-dir|json` |
| PR-mode review | `CALL_TOOL` | `bunx deepsec process --diff <base> --comment-out comment.md` |
| Author custom matcher | `WRITE` | `.deepsec/matchers/<slug>.ts` + inline plugin in `deepsec.config.ts` |
| Validate matcher hit rate | `VALIDATE` | `bunx deepsec scan --matchers <slug>` candidate count |
| Verify and report | `NOTIFY` | `RunMeta`, severity counts, dollar cost, FP rate |
| Stop on budget breach | `TERMINATE` | Refuse unbounded `process` without calibration |

### Tools and instruments
- **Package manager**: `bun` / `bunx` (preferred), `pnpm`, `npm`, `yarn` are interchangeable.
- **CLI commands**: `deepsec init`, `init-project`, `scan`, `process`, `process --diff`, `triage`, `revalidate`, `enrich`, `report`, `export`, `metrics`, `status`, `sandbox <cmd>`.
- **Diff sources for PR mode**: `--diff <ref|range>`, `--diff-staged`, `--diff-working`, `--files <csv>`, `--files-from <path>` (or `-` for stdin).
- **Inspection**: `jq` over `data/<id>/files/**/*.json` for ad-hoc severity / TP queries.
- **Credentials**: `AI_GATEWAY_API_KEY`, `VERCEL_OIDC_TOKEN`, `ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_BASE_URL`, `OPENAI_API_KEY` / `OPENAI_BASE_URL`, `claude login`, `codex login`.
- **Resource files** under `resources/` for setup, scanning, PR review, matchers, triage, config, load on demand.

### Canonical workflow path
1. **Bootstrap** (one time per repo):
   ```bash
   cd <target-repo>
   bunx deepsec init
   cd .deepsec
   bun install
   # Edit .env.local: set AI_GATEWAY_API_KEY=vck_… (or VERCEL_OIDC_TOKEN via `vercel env pull`)
   ```
   Then prompt the coding agent (this skill) to read
   `.deepsec/node_modules/deepsec/SKILL.md` and `.deepsec/data/<id>/SETUP.md`,
   skim `README` / `AGENTS.md` / `CLAUDE.md` and a handful of representative
   files, and replace each section of `data/<id>/INFO.md` (50-100 lines,
   3-5 examples per section, no line numbers, no generic CWE rehash).
2. **Calibrate before any full pass.** The deepsec docs (`getting-started.md`, `vercel-setup.md`, `faq.md`) recommend `--limit 50 --concurrency 5` as the calibration starting point.
   ```bash
   bunx deepsec scan
   bunx deepsec status
   bunx deepsec process --limit 50 --concurrency 5
   ```
   Read the per-batch cost. Extrapolate to full repo. Get the user's explicit go-ahead before the full `process`. If the user names different `--limit` / `--concurrency` values, use theirs.
3. **Full investigation, triage, revalidate, export**:
   ```bash
   bunx deepsec process --concurrency 5
   bunx deepsec triage --severity HIGH
   bunx deepsec revalidate --min-severity HIGH
   bunx deepsec export --format md-dir --out ./findings
   bunx deepsec metrics
   ```
4. **PR mode** (CI gate, scoped to changed files, exit code = 0/1):
   ```bash
   bunx deepsec process \
     --diff origin/${BASE_REF} \
     --comment-out comment.md
   ```
   Wire the two-job CI pattern from `resources/pr-review.md`. Never grant `pull-requests: write` to the job that runs PR-controlled code.
5. **Custom matchers** (close entry-point gaps surfaced in step 3):
   - Read the contract in `.deepsec/node_modules/deepsec/dist/config.d.ts` and the `samples/webapp/matchers/*` examples.
   - Write `.deepsec/matchers/<slug>.ts`, wire it through the inline plugin in `.deepsec/deepsec.config.ts`.
   - Verify hit rate: `bunx deepsec scan --matchers <slug>` should land in 1-20 hits / 1k files (`precise`), 5-100 (`normal`), or roughly the framework entry-point count (`noisy`).
6. **Resume** after any quota stop, network blip, or Ctrl-C: re-run the same command. State is on disk under `.deepsec/data/<id>/`.

### Resource scope
| Scope | Resource target |
|-------|-----------------|
| `CODEBASE` | Target repo source files, framework configs, route directories, `README` / `AGENTS.md` / `CLAUDE.md`. |
| `LOCAL_FS` | `.deepsec/deepsec.config.ts`, `.deepsec/.env.local`, `.deepsec/matchers/`, `.deepsec/data/<id>/{project.json,INFO.md,config.json,files/,runs/,reports/}`, generated `findings/`, `comment.md`, CI workflow files. |
| `PROCESS` | `bunx deepsec scan|process|triage|revalidate|export|metrics|status|sandbox`, `bun install`, optional `vercel link` / `vercel env pull`. |
| `NETWORK` | Anthropic / OpenAI via Vercel AI Gateway (default) or direct provider endpoints; optional Vercel Sandbox microVM control plane. |
| `CREDENTIALS` | `AI_GATEWAY_API_KEY`, `VERCEL_OIDC_TOKEN`, `ANTHROPIC_AUTH_TOKEN`, `OPENAI_API_KEY`, `VERCEL_TOKEN` / `VERCEL_TEAM_ID` / `VERCEL_PROJECT_ID`, `claude` / `codex` subscription tokens. Consume read-only; never echo secrets back to the user or commit them. |
| `MEMORY` | User-stated budget cap, severity floor, and stop conditions for the current session. |

### Preconditions
- Node.js 22+ is available.
- Repo is a git checkout (deepsec uses git history for `revalidate` and `--diff`).
- For any AI command: at least one credential mode is configured *before* the call, or the call is held until one is.
- For `sandbox` mode: Vercel auth is wired; otherwise stay local.
- For unbounded `process` runs on > 500-file repos: a `--limit` calibration pass has produced a cost number the user has acknowledged.

### Effects and side effects
- Creates `.deepsec/` (config, lockfile, scaffolding) and `.deepsec/data/<id>/` (gitignored) inside the target repo.
- Writes `.env.local` (never commit) and may run `vercel link` / `vercel env pull` (writes `.vercel/project.json` + token).
- Spawns long-running AI processes that **cost real money**. Single full scans range from $25 to over $1,200 per the official cost guide and can climb to tens of thousands on very large repos.
- Reads source code; sends snippets to the configured LLM (gateway = zero retention; direct provider = subject to that provider's policy). Never exfiltrates secrets; the gateway key stays outside the worker sandbox in `sandbox` mode.
- May write `.github/workflows/deepsec.yml` (or analogue) when the user asks for a CI gate.
- Edits `deepsec.config.ts` and adds `.deepsec/matchers/*.ts` when authoring matchers.
- Does not commit, push, or open PRs unless the user explicitly authorizes a separate commit step (route via `oma-scm`).

### Guardrails
1. **Never launch an unbounded `process` on a repo whose size you have not measured.** Always run a calibration pass first when file count is unknown or > 500 (deepsec docs recommend `--limit 50 --concurrency 5`; defer to a user-named value if given).
2. **State cost and stopping condition before any AI pass.** Use the published bands (100 files ≈ $25-60, 500 ≈ $130-300, 2,000 ≈ $500-1,200; ×2-3 swing).
3. **Resume, do not reset.** After any network / quota / Ctrl-C interruption, re-run the same command. Never delete `data/<id>/` to "start clean" without explicit user instruction.
4. **`INFO.md` stays short and project-specific.** 50-100 lines, 3-5 examples per section. Name primitives but no line numbers. Skip generic CWE categories; built-in matchers cover those.
5. **For PR/CI gates, keep PR-controlled code in a no-write job.** Never grant `pull-requests: write` to a job that executes PR-controlled `pnpm install` / config-loading. Use the two-job pattern in `resources/pr-review.md`.
6. **Pin actions to full SHAs** in production CI; major-version tags are for examples only.
7. **Never silently drop refusals.** If the agent reports `refused: true`, log it, retry with the other backend, or add the file to `ignorePaths` only when reproducible.
8. **Bias matchers toward `precise` when the bug shape is exact.** Reserve `noisy` for entry-point coverage and tight globs.
9. **Never echo or commit credentials** (`vck_…`, `sk-ant-…`, `sk-…`, OIDC tokens). Treat `.env.local` as secret. Treat `data/` as gitignored by default.
10. **Treat deepsec like an agent with shell access.** Recommend `sandbox` for prompt-injection-prone repos (vendored code, untrusted deps).
11. **Findings need verdicts.** For any HIGH+ surfaced to the user, prefer `revalidate`-tagged verdicts (`true-positive` / `false-positive` / `fixed` / `uncertain`) over raw `process` output.
12. **Do not invent CLI flags.** Anything beyond `resources/scanning.md`'s flag list must be checked against `--help` first.
13. **Ask agent choice before the first paid call.** If the user has not named an agent (`claude` vs `codex`) and `deepsec.config.ts` does not pin `defaultAgent`, ask once with the trade-off clearly stated. Do not also bargain over budget or severity; those are handled via the upstream calibration recommendation (`--limit 50 --concurrency 5` per deepsec docs) and the user-stated `severity_floor`.

## References
- Workspace install + `INFO.md` bootstrap: `resources/setup.md`
- Full scan/process/triage/revalidate/export workflow + cost guide: `resources/scanning.md`
- PR / CI gate via `process --diff` (two-job pattern, exit-code semantics): `resources/pr-review.md`
- Authoring custom matchers (slugs, noise tiers, file globs, plugin wiring): `resources/matchers.md`
- Reading findings, severities, triage / revalidation verdicts, FP cuts: `resources/triage.md`
- `deepsec.config.ts` reference, env vars, plugin order, AI Gateway / Vercel Sandbox auth: `resources/config.md`
- Upstream docs (load only when a resource file points at one):
  - Repo + README: https://github.com/vercel-labs/deepsec
  - Per-topic docs at https://github.com/vercel-labs/deepsec/tree/main/docs (`getting-started`, `reviewing-changes`, `writing-matchers`, `configuration`, `models`, `plugins`, `architecture`, `data-layout`, `vercel-setup`, `supported-tech`, `faq`)
- Shared context loading: `../_shared/core/context-loading.md`
- Shared quality principles: `../_shared/core/quality-principles.md`
