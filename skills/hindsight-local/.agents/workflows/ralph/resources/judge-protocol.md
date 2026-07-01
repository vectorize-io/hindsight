# JUDGE Protocol: Independent Verifier

This protocol governs how the JUDGE phase operates in the ralph workflow.
The verifier MUST be independent from the implementer; evaluate only evidence, not intent.

---

## Core Principles

1. **Evidence-based only**: Every judgment must cite concrete evidence (test output, build log, file path, command result)
2. **No subjective assessment**: "Looks correct" or "should work" is NOT valid evidence
3. **Mechanical verification**: Use the verification method defined in the criterion, not alternative methods
4. **Independent perspective**: Judge as if you did not implement the code; verify what IS, not what was intended

---

## Judge Execution Mode

The judge runs as a **spawned subagent with fresh context** (see ralph.md Step 2.1). It receives only the criteria table, the verification cache records, and this protocol — never the implementer's narration or claims about what was fixed. Independence is structural, not a prompt-level role-play.

- **Input**: judge brief (criteria + cache records + output format)
- **Output**: write the JUDGE result (format below) to memory as `result-judge-{sessionId}-iter{N}.md`
- **Model**: verification is mechanical (run command, check exit code/output); a lower-cost model tier is acceptable where the runtime supports per-agent model selection
- **Inline fallback**: when subagent spawning is unavailable, ralph runs this protocol inline and records the independence downgrade (`ralph.judge-inline-fallback` event). The protocol applies identically in both modes.

---

## Verification Methods

For each criterion, execute the defined verification method:

| Verification Type | How to Execute | PASS Condition | FAIL Condition | Cacheable |
|-------------------|----------------|----------------|----------------|-----------|
| `tests pass` | Run test command via Bash | Exit code 0, all tests pass | Any test failure or exit code != 0 | Yes (heavy) |
| `build succeeds` | Run build command via Bash | Exit code 0, no errors | Build errors present | Yes (heavy) |
| `file exists` | Check file path | File exists at specified path | File not found | No (always fast) |
| `command output` | Run specified command | Output matches expected pattern | Output does not match | Conditional |
| `lint passes` | Run lint command via Bash | Zero errors (warnings OK) | Any lint error | Yes (medium) |
| `type check passes` | Run type-check command | Exit code 0 | Type errors present | Yes (medium) |

---

## JUDGE Result Format

```markdown
## JUDGE Result — Iteration {N}

| Criterion | Status    | Evidence                                                         |
|-----------|-----------|------------------------------------------------------------------|
| C1        | PASS      | `bun test` exit 0, 13/13 passed                                  |
| C2        | FAIL      | `bun build` exit 1, TypeError in Form.tsx:42                     |
| C3        | BLOCKED   | Failed 3x: same import resolution error                          |
| C4        | REGRESSED | previously PASS at iter 1 — `curl :3000/health` now timeouts; docker-compose.yml modified in iter 2 |

verdict: PASS | FAIL
```

### Status Definitions

- **PASS**: Verification method executed successfully, evidence confirms criterion is met
- **FAIL**: Verification method executed, evidence shows criterion is NOT met (and this is not a regression: either first failure or persistent failure with `previous_status != PASS`)
- **REGRESSED**: Verification failed AND the criterion's `previous_status` was `PASS`. This is a distinct signal from FAIL, emitted exactly once on the PASS → FAIL transition. On subsequent failures, the criterion follows the normal FAIL → BLOCKED progression.
- **BLOCKED**: Criterion has failed 3 consecutive times across iterations; no further retries

### Verdict Rules

- `PASS`: ALL criteria are PASS or BLOCKED (no FAIL or REGRESSED remaining)
- `FAIL`: ANY criterion has status FAIL or REGRESSED

---

## Remaining Items (on FAIL verdict)

When verdict is FAIL, output remaining work for REPLAN:

```markdown
remaining:
  - id: C{N}
    reason: "<specific failure evidence>"
    suggested_action: "<concrete next step>"
    fail_count: {N}
    regression: {true|false}            # true when a previously passing criterion now fails
    previous_pass_iteration: {N|null}   # iteration where it last passed (null if never)
```

### Suggested Action Guidelines

- Be specific: "Fix TypeError in Form.tsx:42, `props.onChange` is undefined" not "fix the error"
- Reference exact files and line numbers when available
- If the same failure recurred, suggest a DIFFERENT approach than the previous iteration
- If approaching BLOCKED threshold (fail_count = 2), flag it:
  `Next failure will BLOCK this criterion`

---

## BLOCKED Marking Rules

A criterion is marked BLOCKED when:

1. It has `fail_count >= 3` (failed in 3 consecutive iterations)
2. The same root cause persists despite different approaches

`fail_count` counts **consecutive** failures only: it resets to 0 whenever the criterion passes. An intermittently flaky verification therefore cannot accumulate to BLOCKED across non-consecutive failures — recurring flakiness surfaces as repeated REGRESSED signals instead, which is the correct signal for REPLAN to investigate.

When marking BLOCKED:
- Record the 3 failure evidences for reference
- Do NOT retry in subsequent iterations
- Report in the final summary as unresolved

---

## Verification Execution Order

1. Run all verification commands in parallel when possible
2. Collect all results before producing the JUDGE result
3. Do NOT stop at the first failure; verify ALL criteria every iteration (including criteria with `previous_status == PASS`)
4. Record raw command output as evidence (not summaries)

---

## Regression Detection

A regression is detected when a criterion that was `PASS` in an earlier iteration fails verification in the current iteration.

### Why this matters

Ralph's EXEC phase delegates implementation to ultrawork, which freely modifies shared code (utilities, configs, migrations, dependencies). A PASS in iteration N can be silently invalidated by a change ultrawork makes while fixing other criteria in iteration N+1. Re-verifying every criterion every iteration, and labeling PASS → FAIL transitions explicitly, closes this gap.

### Detection rule

```
For each criterion in current iteration:
  if verification_failed AND previous_status == "PASS":
    status := REGRESSED
    regressed_at_iteration := current_iteration
    # do NOT increment fail_count on the first regression
  elif verification_failed:
    fail_count += 1
    status := BLOCKED if fail_count >= 3 else FAIL
  else:
    status := PASS
    fail_count := 0                 # consecutive semantics: a pass breaks the failure streak
    regressed_at_iteration := null
```

### Evidence to capture for REGRESSED

When emitting a REGRESSED entry, the evidence cell MUST include:

1. The iteration number where the criterion last passed
2. The current verification output showing failure
3. A summary of files modified between the last PASS iteration and now (use `git diff --name-only HEAD~{iterations_since_pass} HEAD`, scoped to the project root)

This gives the REPLAN phase enough context to dispatch a diff-aware diagnosis rather than a from-scratch reimplementation.

---

## Caching for Heavy Verification

Re-verifying every criterion every iteration is correct but can be expensive when verifications include long-running steps (e2e tests, Docker rebuilds, full integration suites). Use the cache rules below to skip re-execution when no relevant files changed since the last successful verification.

### Cache eligibility

A verification is eligible for caching when ALL of:

1. The verification is one of the `Yes (heavy)` or `Yes (medium)` types in the verification table
2. The criterion's previous result was PASS
3. The criterion declares an `affected_paths` glob list (added at criterion definition time, see Step 0.2)
4. No file matching `affected_paths` was modified since the last PASS

### Cache invalidation

The cache for criterion C is invalidated when ANY of:

1. Any file matching C's `affected_paths` was modified in the current iteration (use `git diff --name-only` since last PASS)
2. The criterion's verification command changed
3. More than 3 iterations have passed since the cached result (forced re-verification for staleness)

### Cache record format

Maintain in `session-ralph-{sessionId}.md`:

```markdown
verification_cache:
  - id: C5
    cached_status: PASS
    cached_at_iteration: 3
    cached_evidence: "playwright admin.spec.ts: 8/8 passed in 4m12s"
    affected_paths:
      - "src/admin/**"
      - "src/auth/**"
      - "playwright/admin.spec.ts"
    last_verified_iteration: 3
```

### When NOT to cache

Always re-execute (never cache) when:

- Verification type is `file exists` (always fast, no point caching)
- The criterion is REGRESSED or FAIL in the current snapshot (must re-verify after the fix attempt)
- Caching is disabled for the session: the user's request explicitly asks for full re-verification (e.g., "no cache", "always re-verify"). (A `ralph.judge_cache: false` config key in `.agents/oma-config.yaml` is planned but not yet implemented.) Record the disable reason in `session-ralph-{sessionId}.md`.

### Cost-benefit guideline

Apply caching only when the verification takes >30s. Below that threshold, the bookkeeping overhead is not worth the saved time, and uniform re-execution keeps the protocol simpler.
