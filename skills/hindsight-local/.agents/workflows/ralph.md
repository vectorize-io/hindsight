---
name: ralph
description: Ralph - persistent self-referential execution loop wrapping ultrawork with a spawned independent judge
disable-model-invocation: true
---

# MANDATORY RULES: VIOLATION IS FORBIDDEN

- **Response language follows `language` setting in `.agents/oma-config.yaml` if configured.**
- **NEVER skip phases.** Execute from Phase 0 in order. Explicitly report completion of each phase to the user before proceeding to the next.
- **You MUST use MCP tools throughout the entire workflow.** This is NOT optional.
  - Use code analysis tools (`get_symbols_overview`, `find_symbol`, `find_referencing_symbols`, `search_for_pattern`) for code exploration.
  - Use memory tools (read/write/edit) for progress tracking.
  - Memory path: configurable via `memoryConfig.basePath` (default: `.serena/memories`)
  - Tool names: configurable via `memoryConfig.tools` in `.agents/mcp.json`
  - Do NOT use raw file reads or grep as substitutes. MCP tools are the primary interface for code and memory operations.
- **This workflow does NOT stop until all completion criteria pass or safeguards trigger.**
- **Follow the context-loading guide.** Read `.agents/skills/_shared/core/context-loading.md` and load only task-relevant resources.

---

## Vendor Detection

Before starting, determine your runtime environment by following `.agents/skills/_shared/core/vendor-detection.md`.
The detected vendor determines how ultrawork spawns agents internally.

---

## Phase 0: INIT (DO NOT SKIP)

### Step 0.1: Load Prerequisites

1. Read `.agents/skills/_shared/core/context-loading.md` for resource loading strategy.
2. Read `.agents/skills/_shared/runtime/memory-protocol.md` for memory protocol.
3. Read `.agents/workflows/ralph/resources/judge-protocol.md` for JUDGE rules.
4. Read `.agents/skills/_shared/runtime/event-spec.md` for the L1 event protocol and `oma state:emit` (used by the EXEC checkpoint in Step 1.2).

### Step 0.2: Define Completion Criteria

Analyze the user's request and define **verifiable** completion criteria. Each criterion MUST have:

```markdown
criteria:
  - id: C{N}
    description: "<what to achieve>"
    verification: "<how to verify — test result, build output, file existence, command output>"
    status: PENDING
    fail_count: 0                   # consecutive failures only — resets to 0 on PASS
    previous_status: null           # last non-null status from prior iteration
    regressed_at_iteration: null    # iteration number when PASS → FAIL transition was detected
    affected_paths: []              # optional glob list — only set when verification takes >30s
                                    # used by judge-protocol's cache rules; see judge-protocol.md § "Caching for Heavy Verification"
```

**Rules:**
- Every criterion must be mechanically verifiable (test pass, build success, file exists, command output)
- Reject subjective criteria ("looks good", "feels right"). Ask the user to rephrase.
- Present criteria to the user for confirmation before proceeding

### Step 0.3: Initialize Session

1. Generate a `sessionId` (`{YYYYMMDD-HHmmss}` timestamp). All ralph memory files for this run are session-scoped with this suffix, per memory-protocol session-scoped naming. Never write to an unsuffixed `session-ralph.md` — consecutive ralph runs must not overwrite each other.
2. Set `max_iterations: 5` (default safeguard)
3. Set `current_iteration: 0`
4. **Load prior-session context** (cross-session memory):
   1. Use the memory list tool to find previous `session-ralph-*.md` files. If any exist, read the most recent one and extract: final criteria statuses, BLOCKED items with their failure evidences, and any safeguard trigger.
   2. If `lessons-learned.md` exists in the memory base path, read it.
   3. If any current criterion overlaps a previously BLOCKED item, re-confirm with the user before proceeding: present the prior failure evidence and ask whether to retry it (carrying that evidence as context for EXEC) or pre-mark it BLOCKED for this session.
5. Record session start using memory write tool:
   - Create `session-ralph-{sessionId}.md` in the memory base path
   - Include: session start time, user request summary, completion criteria, max_iterations, and prior-session findings loaded in step 4 (or `none`)

---

## Phase 1: EXEC

// turbo

### Step 1.1: Prepare Ultrawork Input

Compose the ultrawork input based on current iteration:

- **Iteration 1**: Full user request with all PENDING criteria
- **Iteration 2+**: REMAINING (FAIL + REGRESSED) criteria from previous JUDGE result, with:
  - Previous JUDGE results as context (what failed and why)
  - Suggested actions from JUDGE
  - Already-PASSED criteria excluded from **implementation scope** (do not re-implement), but they remain in **JUDGE scope** (will be re-verified to detect regressions)

### Step 1.2: Execute Ultrawork

**EXEC-entry checkpoint (MANDATORY — emit before delegating).** This records, in the auditable L1 event log, that this iteration delegates to the full ultrawork workflow. A run without this event is a non-compliant run.

```bash
oma state:emit "decision.made" '{"subject":"ralph.exec-delegated","decision":"Delegate this iteration to the full ultrawork 5-phase workflow.","rationale":"Ralph EXEC must run ultrawork in full; abridging, substituting, or skipping phases for cost/stability/time reasons is forbidden without explicit user approval."}'
oma state:verify --workflow ralph --checkpoint exec-delegated
```

Delegate to the ultrawork workflow:

1. Read and follow `.agents/workflows/ultrawork.md` step by step.
2. Pass the prepared input as the task description.
3. Ultrawork handles all vendor-specific agent spawning internally.
4. Wait for ultrawork to complete all 5 phases (PLAN, IMPL, VERIFY, REFINE, SHIP).
5. **Do NOT abridge ultrawork.** If you believe the environment (subagent instability, cost, time) warrants reducing fan-out or collapsing phases, STOP and ask the user first. Single-judgment substitution of ultrawork's structure is forbidden — see the Anti-Circumvention gate in Step 1.3.

### Step 1.3: Verify EXEC Artifacts (Anti-Circumvention Gate)

**Prose instructions ("run ultrawork in full") are advisory and can be rationalized away. This gate verifies the work mechanically — by its artifacts, not by your own narration.** Ultrawork's 5 phases each leave a durable trace; a single-agent shortcut cannot produce them without actually doing the work.

Run the deterministic verifier from the repo root:

```bash
oma ralph:verify --json --session {sessionId} --newer-than {iteration_start_iso}
```

- `--session` scopes the plan artifact to this iteration's session id; `--newer-than` (this iteration's EXEC start time, ISO-8601) excludes stale artifacts from earlier iterations. Omit either when unknown.
- The command checks the artifact table below, prints a structured result (`ok`, `checks`, `missing`, `remediation`), and exits non-zero on failure. On failure it also appends a `gate.failed` L1 event automatically.
- **The JSON verdict IS the gate result.** Do NOT substitute your own narration for it, and do NOT proceed on a non-zero exit.
- **Manual fallback** (only when the `oma` CLI is unavailable): check, using memory read / file existence tools, that the just-completed iteration produced ALL of the artifacts below. Resolve `{memBase}` from `memoryConfig.basePath` (default `.serena/memories`).

| # | Artifact | Proves phase ran |
|---|----------|------------------|
| A1 | `{memBase}/session-ultrawork.md` with this iteration's phase-completion records | PLAN + gate progression |
| A2 | `.agents/results/plan-{sessionId}.json` | PLAN produced a real task breakdown |
| A3 | `{memBase}/result-qa*.md` or `.agents/results/result-qa*.md` (VERIFY) | **a distinct QA agent ran** — absent if IMPL was the only spawn. CLI fallback writes `result-qa-agent*` to `{memBase}`; Claude-native `qa-reviewer` writes `result-qa*` to `.agents/results/` |
| A4 | `{memBase}/result-debug*.md` or `.agents/results/result-debug*.md` (REFINE) | **a distinct Debug agent ran** — same naming split (`debug-investigator` on the native path) |

**Decision:**

- **`ok: true` (exit 0)** → ultrawork ran in full. Proceed to Step 1.4.
- **`ok: false` (exit 1, `missing` non-empty)** → treat EXEC as **NOT performed** (the iteration was abridged to implementation-only, regardless of what the EXEC narration claims). Do NOT advance to JUDGE as if work completed. Instead:
  1. Record the violation in `session-ralph-{sessionId}.md`: `exec-circumvention detected at iteration {N}: missing {artifact}`.
  2. Emit the audit event:
     ```bash
     oma state:emit "decision.made" '{"subject":"ralph.exec-circumvention","decision":"EXEC artifacts incomplete — ultrawork did not run in full.","rationale":"Required VERIFY/REFINE agent result files are absent; the iteration was abridged."}'
     ```
  3. STOP and report to the user that ultrawork was not executed in full, citing the missing artifact. Ask whether to re-run the iteration in full or to explicitly authorize a reduced-scope run. Do NOT silently retry with the same abridged approach.

> **REFINE skip exception**: ultrawork permits skipping REFINE for trivial tasks (< 50 lines, see ultrawork `REFINE_GATE` skip conditions). If REFINE was legitimately skipped, A4 may be absent — but `session-ultrawork.md` MUST record the documented skip reason. "No A4 and no recorded skip reason" is a circumvention, not a skip. `oma ralph:verify` implements this rule: a recorded skip reason reports A4 as `skip-recorded` (passing), an unrecorded absence reports `missing` (failing).

### Step 1.4: Record EXEC Completion

1. Increment `current_iteration`
2. Use memory edit tool to record iteration start in `session-ralph-{sessionId}.md`

---

## Phase 2: JUDGE

### Step 2.1: Independent Verification (Spawned Judge)

**The judge is a separate agent with fresh context — not a role the orchestrator plays.** The orchestrator that drove EXEC shares context with the implementation and cannot self-judge without rationalization risk. Spawning is the default; inline judging is a recorded exception.

1. **Compose the judge brief.** It contains ONLY:
   - The criteria table: id, description, verification method, previous_status, fail_count, affected_paths
   - The verification cache records from `session-ralph-{sessionId}.md` (if any)
   - The required output format (Step 2.2) and a pointer to `.agents/workflows/ralph/resources/judge-protocol.md`
   - Do NOT include EXEC narration, implementation summaries, or any claim about what was fixed. The judge verifies what IS, not what was intended.
2. **Spawn the judge via Per-Agent Dispatch** (see Vendor Detection):
   - **If Claude Code and target vendor is Claude**: `Agent(subagent_type="qa-reviewer", prompt="<judge brief>. Follow .agents/workflows/ralph/resources/judge-protocol.md. Execute every verification command and write the JUDGE result to memory as result-judge-{sessionId}-iter{N}.md.")`
   - **Otherwise, or when native dispatch is unavailable**: `oma agent:spawn qa-agent "<judge brief>" {sessionId}`
   - Verification is mechanical (run command, check exit code/output) — a lower-cost model tier is acceptable where the runtime supports per-agent model selection.
3. **Wait for `result-judge-{sessionId}-iter{N}.md`**, then read it as the JUDGE result.
4. **Inline fallback (exception)**: only if subagent spawning is unavailable in the current runtime, perform the verification inline. Record `judge-inline-fallback at iteration {N}` in `session-ralph-{sessionId}.md` and emit:
   ```bash
   oma state:emit "decision.made" '{"subject":"ralph.judge-inline-fallback","decision":"Run JUDGE inline in the orchestrator context.","rationale":"Subagent spawning unavailable in this runtime; judge independence is downgraded for this iteration."}'
   ```

For **EVERY criterion regardless of current status** (including PASS from prior iterations), the judge executes the verification method defined in Phase 0:

- Run tests, then check pass/fail count
- Run build, then check exit code
- Check file existence and verify path
- Run specific commands, then check output

**Why re-verify PASS criteria**: ultrawork modifies shared code (utils, configs, migrations, dependencies). A PASS in iteration N may regress in iteration N+1 when fixing other criteria. Without re-verification, "DONE" can ship silent regressions.

**Heavy verification caching**: For verifications that take >30 seconds (e2e tests, integration suites), apply the caching rules in `judge-protocol.md` § "Caching for Heavy Verification" to skip re-runs when no relevant files changed.

**Follow `.agents/workflows/ralph/resources/judge-protocol.md` for the full protocol.**

### Step 2.2: Produce JUDGE Result

Output the JUDGE result in this exact format:

```markdown
## JUDGE Result — Iteration {N}

| Criterion | Status    | Evidence                                                |
|-----------|-----------|---------------------------------------------------------|
| C1        | PASS      | <concrete evidence>                                     |
| C2        | FAIL      | <concrete evidence of failure>                          |
| C3        | BLOCKED   | <failed 3x: reason>                                     |
| C4        | REGRESSED | previously PASS at iter N — now FAIL: <evidence + diff> |

verdict: PASS | FAIL
```

If verdict is FAIL, also output:

```markdown
remaining:
  - id: C{N}
    reason: "<why it failed>"
    suggested_action: "<what to try next>"
    fail_count: {N}
    regression: true | false        # true if status is REGRESSED
    previous_pass_iteration: {N}    # only when regression: true
```

### Step 2.3: Apply JUDGE Result

Before updating any criterion, capture the current `status` into `previous_status`. Then apply the transition rules in order:

1. **Verification passed** → `PASS`. Reset `fail_count` to 0 and `regressed_at_iteration` to null (`fail_count` tracks **consecutive** failures only; a pass breaks the streak).
2. **Verification failed AND `previous_status == PASS`** → `REGRESSED`. Set `regressed_at_iteration: {current_iteration}`. Do NOT increment `fail_count` on the first regression; regression is treated as a distinct first-class signal, not a normal failure streak. Subsequent consecutive failures of the same criterion follow rules 3-4.
3. **Verification failed AND not a regression AND `fail_count < 3`** → `FAIL`. Increment `fail_count`.
4. **Verification failed AND `fail_count >= 3`** → `BLOCKED`.

**Decision Gate impact**:
- `REGRESSED` is treated as `FAIL` for verdict computation (verdict becomes FAIL, REPLAN triggers).
- `REGRESSED` is NOT counted toward "DONE"; only `PASS` and `BLOCKED` count.

---

## Phase 2 → Decision Gate

Evaluate the JUDGE result:

### → DONE (All criteria PASS or BLOCKED)

If all criteria are either PASS or BLOCKED:

1. **If any BLOCKED exists**: Report partial completion with BLOCKED items listed
2. **If all PASS**: Report full completion
3. Use memory edit tool to record final results in `session-ralph-{sessionId}.md`
4. Output completion summary:
   ```
   ## Ralph Complete — Iteration {N}/{max}

   PASSED: C1, C2, ...
   BLOCKED: C3 (if any)

   Total iterations: {N}
   ```
5. Workflow ends.

### → REPLAN (Any criterion is FAIL or REGRESSED)

If any criterion has status FAIL or REGRESSED, proceed to Phase 3.

### → SAFEGUARD (max_iterations reached)

If `current_iteration >= max_iterations`:

1. Force stop regardless of FAIL criteria
2. Report partial completion:
   ```
   ## Ralph Safeguard — Max Iterations Reached ({max})

   PASSED: C1, ...
   FAILED: C2, ... (still unresolved)
   BLOCKED: C3, ... (if any)

   Recommendation: Review FAILED criteria manually or increase max_iterations.
   ```
3. Use memory edit tool to record safeguard trigger in `session-ralph-{sessionId}.md`
4. Workflow ends.

---

## Phase 3: REPLAN

// turbo

### Step 3.1: Extract Remaining Work

From the JUDGE result, collect criteria with status `FAIL` or `REGRESSED`. Treat the two classes separately:

1. **FAIL** (first-time or persistent failures): list each with its reason and suggested_action
2. **REGRESSED** (previously PASS, now FAIL): list each with previous-pass iteration, the inter-iteration diff that likely caused the regression, and a regression-specific suggested_action.
3. Include previous iteration's JUDGE evidence as context
4. Explicitly state which criteria are PASS (do not re-implement, but do not exclude from next JUDGE either)
5. Explicitly state which criteria are BLOCKED (do not retry)

### Step 3.2: Narrow Scope

Compose a focused task description containing the remaining work, separating regressions from first-fail items so ultrawork's reasoning differs:

```markdown
## Ralph Iteration {N+1} — Remaining Work

### Already Complete (DO NOT re-implement; will be re-verified by JUDGE)
- C1: <description> PASS

### Blocked (DO NOT retry)
- C3: <description> BLOCKED (failed 3x)

### Regressed (was passing — diagnose what broke it; minimal fix that preserves recent changes)
- C4: <description>
  - Last passed at: iteration {N}
  - Failed at: iteration {current}
  - Files changed since last pass: <list of modified paths>
  - Failure evidence: <evidence>
  - Suggested action: diff-aware diagnosis — identify which change in the listed files broke C4, fix that specifically without reverting the criterion that change was made for

### To Fix (first-time or persistent failures)
- C2: <description>
  - Previous failure: <evidence>
  - Suggested action: <action>
```

**Why separate Regressed from To Fix**: ultrawork prompts that frame work as "fix from scratch" vs "diagnose a regression" produce different reasoning paths. Regressed items should trigger diff-based investigation, not greenfield re-implementation.

### Step 3.3: Loop Back

1. Use memory edit tool to record REPLAN in `session-ralph-{sessionId}.md`
2. Return to **Phase 1: EXEC** with the narrowed scope

---

## Summary

```
Phase 0: INIT → Define criteria, load prior sessions, initialize session
    ↓
Phase 1: EXEC → Run ultrawork (full or narrowed scope)
    ↓
Phase 2: JUDGE → Spawned fresh-context judge verifies each criterion
    ↓
Decision: DONE? → End
          SAFEGUARD? → Force end
          FAIL? → Phase 3
    ↓
Phase 3: REPLAN → Extract remaining, narrow scope
    ↓
    └──→ Phase 1 (loop)
```

| Phase   | Purpose                    | Key Action                        |
|---------|----------------------------|-----------------------------------|
| INIT    | Define success criteria     | Verifiable criteria + prior-session load + session init |
| EXEC    | Implementation             | Delegate to ultrawork             |
| JUDGE   | Independent verification   | Spawned judge; evidence-based pass/fail per criterion |
| REPLAN  | Scope narrowing            | Extract FAIL + REGRESSED items, separated by class |
