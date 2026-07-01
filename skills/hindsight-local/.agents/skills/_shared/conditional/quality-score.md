# Quality Score Continuum

Replaces binary PASS/FAIL gate evaluation with a **continuous quantitative score** (0-100).
Inspired by autoresearch's val_bpb metric: objective, comparable, and trackable over time.

---

## Score Dimensions

| Dimension | Weight | Measurement Method | Measurer |
|-----------|--------|--------------------|----------|
| **Correctness** | 0.30 | Test pass rate (passed / total) | Agent runs `test` via Bash |
| **Security** | 0.25 | OWASP checklist completion rate | QA Agent review |
| **Performance** | 0.15 | No regression vs baseline (estimate) | Agent or QA estimate |
| **Coverage** | 0.15 | Test coverage % from tool output | Agent runs coverage via Bash |
| **Consistency** | 0.15 | Lint + type errors (100 - error_count, min 0) | Agent runs lint/type-check via Bash |

### Composite Score Formula

```
composite = (correctness * 0.30) + (security * 0.25) + (performance * 0.15)
          + (coverage * 0.15) + (consistency * 0.15)
```

---

## Measurement Protocol

### How to Measure (Practical)

Agents with **Bash** tool can measure directly:

```bash
# Correctness: parse test output
npm test 2>&1 | tail -5          # or: uv run pytest -q
# → extract passed/failed counts → score = (passed / total) * 100

# Coverage: parse coverage output
npm run coverage 2>&1 | grep "All files"   # or: uv run pytest --cov
# → extract % → score = coverage_percent

# Consistency: count lint + type errors
npm run lint 2>&1 | grep -c "error"        # or: uv run ruff check
npm run type-check 2>&1 | grep -c "error"
# → score = max(0, 100 - error_count)
```

**When automated tools are unavailable** (no test suite, no lint config):
- Agent estimates the dimension based on code review (0-100)
- Must note `(estimated)` next to the score
- Estimated scores carry lower weight in delta decisions (see below)

### When to Measure

Quality Score is measured **on demand**, not at every step. Load `quality-score.md` only at these checkpoints:

| Checkpoint | Trigger | Measurer |
|-----------|---------|----------|
| IMPL baseline | After implementation complete, before VERIFY | Orchestrator (inline) or impl agent |
| Post-VERIFY | After QA verification complete | QA Agent |
| Post-REFINE | After refinement complete | Debug Agent or Orchestrator |
| Final | Before SHIP_GATE | QA Agent |

---

## Score Thresholds

| Range | Grade | Gate Decision |
|-------|-------|---------------|
| 90-100 | A | PASS, proceed immediately |
| 75-89 | B | CONDITIONAL PASS, proceed with noted improvements |
| 60-74 | C | FAIL, must improve before proceeding |
| 0-59 | D | HARD FAIL, rollback and re-plan required |

---

## Keep/Discard Rule

Changes are evaluated by their **impact on the score**, not just by whether they pass review.

```
IF score_after >= score_before:
    KEEP change
ELSE IF (score_before - score_after) < 5:
    REVIEW (minor regression, justify in experiment ledger)
ELSE:
    DISCARD change (revert and try alternative)
```

### Delta Recording

Every scored change is recorded in the Experiment Ledger (see `experiment-ledger.md`).
Record via memory protocol: `[EDIT]("experiment-ledger.md", append row)`.

---

## Score Record Format

```markdown
### Quality Score @ {PHASE}_{checkpoint}
| Dimension | Score | Detail |
|-----------|-------|--------|
| Correctness | 85 | 17/20 tests pass |
| Security | 90 | No CRITICAL/HIGH, 1 MEDIUM |
| Performance | 75 | (estimated) no regression observed |
| Coverage | 70 | 70% line coverage |
| Consistency | 95 | 0 lint errors, 1 type warning |
| **Composite** | **83.5** | Grade: B |
```

---

## Dimension Customization (Optional)

Projects can override weights in `.agents/config/quality-score.yaml`:

```yaml
weights:
  correctness: 0.25
  security: 0.35
  performance: 0.10
  coverage: 0.15
  consistency: 0.15
thresholds:
  pass: 85
  hard_fail: 60
```

If config file is absent, use the defaults defined in this document.

---

## Integration Points

| Component | How It Uses Quality Score |
|-----------|--------------------------|
| **Phase Gates** | Gate criteria reference composite score threshold |
| **Experiment Ledger** | Records score delta per experiment |
| **Exploration Loop** | Compares scores across alternative approaches |
| **Session Metrics** | Tracks score progression through session |
| **Lessons Learned** | Discarded experiments (delta <= -5) auto-feed lessons |
