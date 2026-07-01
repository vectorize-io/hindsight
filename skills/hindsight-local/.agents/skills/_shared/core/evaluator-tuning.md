# Evaluator Tuning Protocol

QA prompts do not work well out of the box. Reliable evaluation requires
iterative refinement based on observed judgment errors.
(ref: Anthropic harness design, "several rounds of development loop necessary")

This protocol is **semi-automated**: collection is automatic, analysis and patching
require human review via `oma retro`.

---

## Tuning Loop

```
Sessions accumulate EA events in session-metrics.md
    ↓
[1] Trigger: EA >= 30 on rolling 3-session window
    ↓
[2] `oma retro` aggregates EA events and generates tuning report
    ↓
[3] Report categorizes errors by type and suggests specific patches
    ↓
[4] User reviews and approves patches
    ↓
[5] Patches applied to QA checklist.md / execution-protocol.md
    ↓
[6] Validation: Next 3 sessions, check if same error type recurs
    ↓ Recurs → re-enter at [2] with stronger fix
    ↓ Resolved → mark as confirmed in tuning log
```

---

## Patch Targets

| Error Pattern | Patch Target | Example |
|--------------|-------------|---------|
| Missed bug category | QA `checklist.md`: add check item | "Race condition in concurrent writes" |
| Wrong severity | QA `execution-protocol.md`: add calibration rule | "Auth issues are always CRITICAL" |
| Missed stub | QA `checklist.md`: runtime verification section | "Check upload actually processes file" |
| False positive pattern | QA `execution-protocol.md`: add exclusion | "Unused imports in test files are OK" |
| Inconsistent depth | QA `execution-protocol.md`: difficulty link | "Complex tasks require full audit" |

---

## Tuning Log

Maintained in this file. Record each tuning action:

```markdown
### {YYYY-MM-DD}: {one-line summary}
- **Error observed**: {what QA got wrong, with EA event type}
- **Root cause**: {why the prompt allowed this}
- **Patch applied**: {file + change description}
- **Validation**: {pending | confirmed in session {id} | failed — re-tuned}
```

---

## Tuning from Success

When `good_catch` events accumulate (>= 5 in rolling window):

1. Identify which QA instruction or checklist item led to the catch
2. Generalize: Can this pattern apply to other domains?
3. If yes: Propose addition to `common-checklist.md`
4. Record in tuning log as positive reinforcement

This prevents tuning drift toward pure skepticism; QA must also know what it does well.

---

## Integration Points

| System | Role |
|--------|------|
| `session-metrics.md` | EA event source (automatic collection) |
| `oma retro` | Aggregation + tuning report generation (semi-automated) |
| `lessons-learned.md` | QA Evaluation Lessons for persistent patterns |
| QA `checklist.md` | Primary patch target for missed checks |
| QA `execution-protocol.md` | Patch target for process/severity issues |
| `common-checklist.md` | Propagation target for generalized good patterns |
