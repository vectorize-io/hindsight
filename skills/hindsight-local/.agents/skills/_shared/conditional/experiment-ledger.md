# Experiment Ledger

An automatic record of every significant change attempt and its measurable outcome.
Inspired by autoresearch's git-commit-as-experiment-log pattern.

---

## Ledger Location

The ledger follows the **memory protocol** (see `memory-protocol.md`):

- **MCP mode** (Serena): `[WRITE]("experiment-ledger.md", ...)` → `{memoryConfig.basePath}/experiment-ledger.md`
- **File-based mode** (Claude protocol): `.agents/results/experiment-ledger.md`

Both modes use the same format. The orchestrator creates the ledger; agents append via memory tools.

---

## Ledger Format

```markdown
# Experiment Ledger — Session {SESSION_ID}
Started: {ISO timestamp}
Request: "{original user request, first 100 chars}..."

## Experiments

| # | Phase | Agent | Hypothesis | Score Before | Score After | Delta | Decision | Files Changed |
|---|-------|-------|-----------|-------------|------------|-------|----------|---------------|
| 1 | IMPL | backend | REST API with pagination | — | 72 | — | BASELINE | 3 |
| 2 | VERIFY | qa | Add input validation | 72 | 78 | +6 | KEEP | 2 |
| 3 | REFINE | debug | Extract shared util | 78 | 80 | +2 | KEEP | 4 |
| 4 | REFINE | debug | Redis caching layer | 80 | 76 | -4 | DISCARD | 3 |
| 5 | REFINE | backend | Simpler in-memory cache | 80 | 84 | +4 | KEEP | 1 |

## Summary
- Total experiments: 5
- Kept: 3 (60%)
- Discarded: 1 (20%)
- Baseline: 1 (20%)
- Net score improvement: +12 (72 → 84)
- Most effective agent: backend (+4 avg delta)
```

---

## Recording Protocol

### What Constitutes an "Experiment"

An experiment is recorded when:
1. A discrete logical change is applied (not individual line edits)
2. A quality score can be measured before and after
3. A keep/discard decision is made per `quality-score.md`

Do NOT record: trivial formatting, changes with no measurable impact, PLAN phase.

### Recording Steps

1. Note current quality score (or `—` for first baseline)
2. Apply change
3. Measure new quality score
4. Calculate delta: `score_after - score_before`
5. Apply Keep/Discard rule from `quality-score.md`
6. Append row via memory tools: `[EDIT]("experiment-ledger.md", append row)`

### Who Records

See `memory-protocol.md` → "Experiment Tracking" section for recorder assignments.

---

## Session-End Analysis

At session completion, the orchestrator generates a summary:

```markdown
## Ledger Analysis

### Score Trajectory
IMPL: {score} → VERIFY: {score} → REFINE: {score} → Final: {score}

### Top Improvements (by delta)
1. Experiment #{N}: {hypothesis} → +{delta}

### Failed Experiments (learning opportunities)
1. Experiment #{N}: {hypothesis} → {delta} — Root Cause: {why it failed}

### Agent Effectiveness
| Agent | Experiments | Avg Delta | Keep Rate |
|-------|------------|-----------|-----------|
| backend | 3 | +4.0 | 67% |
```

---

## Integration with Lessons Learned

Discarded experiments with **delta <= -5** auto-generate lesson candidates at session end.

Format (matches `lessons-learned.md` RCA format):

```markdown
### {YYYY-MM-DD}: {agent-type} - {hypothesis} (DISCARDED, delta: {delta})
- **Problem**: {what was attempted}
- **Root Cause**: {why score decreased — which dimension regressed and why}
- **Lesson**: {what to avoid or do differently next time}
- **Source**: Experiment Ledger #{experiment_number}, Session {session_id}
```

The orchestrator appends these to the relevant domain section in `lessons-learned.md`.

---

## Integration Points

| Component | How It Uses Experiment Ledger |
|-----------|------------------------------|
| **Quality Score** | Provides score measurements for delta calculation |
| **Exploration Loop** | Records parallel experiments and winner selection |
| **Session Metrics** | Experiment count and keep rate in session summary |
| **Lessons Learned** | DISCARD experiments (delta <= -5) auto-generate lessons |
| **Memory Protocol** | Ledger uses same read/write tools as other memory files |
