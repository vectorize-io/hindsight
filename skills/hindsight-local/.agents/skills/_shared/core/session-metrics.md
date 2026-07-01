# Session Metrics & Clarification Debt Tracking

Tracks per-session agent performance metrics, with emphasis on **Clarification Debt (CD)**, the cost of unclear requirements, scope creep, and charter violations.

---

## Why Track This?

Agents that frequently require re-direction consume more tokens and user time than agents that "get it right" the first time. CD tracking enables:

1. **Pattern Detection**: Identify which agents/task-types cause the most re-work
2. **Prompt Improvement**: Data-driven refinement of task descriptions
3. **Escalation Triggers**: Automatic RCA requirements when thresholds exceeded

---

## Clarification Debt (CD) Scoring

| Event Type | Points | Description |
|------------|--------|-------------|
| `clarify` | +10 | Simple clarification question (expected for MEDIUM uncertainty) |
| `correct` | +25 | Intent misunderstanding requiring direction change |
| `redo` | +40 | Scope/charter violation requiring rollback and restart |
| `blocked` | +0 | Agent correctly stopped and asked (this is GOOD behavior) |

### Scoring Modifiers

| Condition | Modifier |
|-----------|----------|
| Charter not read before action | +15 |
| Allowlist violation (file outside scope) | +20 |
| Same error type repeated in session | x1.5 |

---

## Thresholds & Actions

| Threshold | Scope | Action |
|-----------|-------|--------|
| CD >= 50 | Single session | **MANDATORY**: Add RCA to `lessons-learned.md` |
| CD >= 30 | Same agent, 3 consecutive sessions | **REVIEW**: Examine agent prompt template |
| CD >= 80 | Single session | **ESCALATE**: Halt session, request user re-specification |
| `redo` count >= 2 | Single session | **PAUSE**: Orchestrator requests explicit scope confirmation |

---

## Session Log Format

Orchestrator maintains this log in `.serena/memories/session-metrics.md` during execution.

```markdown
## Session: {SESSION_ID}
Started: {ISO timestamp}
Request: "{original user request, first 100 chars}..."

### Events

| Turn | Agent | Event | Points | Detail |
|------|-------|-------|--------|--------|
| 5 | backend | correct | 25 | Changed from REST to GraphQL per user correction |
| 12 | frontend | clarify | 10 | Asked about dark mode preference |
| 18 | backend | redo | 40 | Auth approach rejected, restarting with OAuth |

### Summary
- Total CD: 75
- Agents: backend (65), frontend (10)
- Threshold breached: YES (CD >= 50)
- RCA Required: YES
```

---

## Event Recording Protocol

### For Orchestrator

When user sends a correction/clarification during session:

1. **Classify** the event type:
   - Is user answering a question agent asked? → `clarify`
   - Is user correcting a misunderstanding? → `correct`
   - Is user rejecting work and asking for restart? → `redo`

2. **Record** via MCP memory:
   ```
   [EDIT]("session-metrics.md", append event row)
   ```

3. **Check threshold** after each event:
   - If CD >= 80: Pause and request re-specification
   - If `redo` >= 2: Request explicit scope confirmation

### For QA Agent (Post-Session)

At session end, if total CD >= 50:

1. **Generate RCA** with this format:
   ```markdown
   ### {date}: {agent} - CD threshold breach ({score} points)
   - **Problem**: {what triggered the corrections}
   - **Root Cause**: {why the misunderstanding occurred}
   - **Fix Applied**: {how it was resolved}
   - **Prevention**: {prompt/process change to prevent recurrence}
   ```

2. **Append** to `lessons-learned.md` in the relevant domain section

---

## Integration Points

| Component | How It Uses Session Metrics |
|-----------|----------------------------|
| **Orchestrator** | Records events, checks thresholds, triggers pauses |
| **QA Agent** | Reviews session metrics, generates RCA if needed |
| **Dashboard** | Displays real-time CD score (optional) |
| **Retro Command** | Aggregates CD across sessions for trend analysis |

---

## Example: Healthy vs Unhealthy Session

### Healthy Session (CD = 10)
```
Turn 3: frontend asked about icon library preference → clarify (+10)
Turn 15: All tasks completed successfully
Total CD: 10
```

### Unhealthy Session (CD = 95)
```
Turn 2: backend assumed REST, user wanted GraphQL → correct (+25)
Turn 8: backend used wrong auth method → correct (+25)
Turn 12: frontend built wrong layout → redo (+40)
Turn 14: Charter not checked before redo → modifier (+15, but capped)
Total CD: 95 → RCA REQUIRED
```

---

## Quality Score Tracking (Extension)

When Quality Score measurement is active (see `quality-score.md`), the session log includes score progression.

### Score Progression (appended to session log)

```markdown
### Quality Score Progression

| Checkpoint | Phase | Composite | Grade | Delta |
|-----------|-------|-----------|-------|-------|
| Baseline | IMPL end | 72 | C | n/a |
| Post-VERIFY | VERIFY end | 78 | B | +6 |
| Post-REFINE | REFINE end | 84 | B | +6 |
| Final | SHIP | 86 | B | +2 |
```

### Experiment Summary (appended to session log)

```markdown
### Experiment Summary
- Total experiments: {N}
- Kept: {N} ({%})
- Discarded: {N} ({%})
- Exploration rounds: {N} (max 2 per session)
- Net score improvement: {start} → {final} (delta: {+N})
```

This data is sourced from the Experiment Ledger at session end (see `experiment-ledger.md`).

---

## Metrics Retention

- **Active session**: `.serena/memories/session-metrics.md`
- **Completed sessions**: Archived to `.serena/memories/archive/metrics-{date}.md`
- **Retention**: 30 days (configurable)
- **Aggregation**: `oma stats` command summarizes trends

---

## Evaluator Accuracy Tracking

QA agents improve only when their judgment errors are tracked.
Unlike CD (tracked in real-time), Evaluator Accuracy (EA) is a
**retrospective metric**; most errors are discovered after the session ends.

### Accuracy Events

| Event | Points | When Discovered |
|-------|--------|-----------------|
| `false_negative` | +30 | Next session or production: bug that QA missed |
| `false_positive` | +15 | During session: impl agent disputes QA finding successfully |
| `severity_mismatch` | +10 | During session or retro: wrong severity assigned |
| `missed_stub` | +20 | During session: runtime verification catches display-only feature |
| `good_catch` | -10 | During session: QA caught non-obvious bug (reward signal) |

### Recording

- `false_positive`, `missed_stub`, `good_catch`: Recorded during session by Orchestrator
- `false_negative`, `severity_mismatch`: Recorded retroactively via `oma retro` or next session discovery

### Evaluator Accuracy Score (EA)

Calculated on a **rolling 3-session window**, not per single session:

```
EA = sum(accuracy_event_points across last 3 sessions)
```

| Threshold | Action |
|-----------|--------|
| EA >= 30 | **TUNING SUGGESTED**: `oma retro` flags QA patterns for review |
| EA >= 50 | **TUNING REQUIRED**: Review and update QA execution-protocol.md |
| `false_negative` >= 3 across window | **CHECKLIST UPDATE**: Add detection pattern to QA checklist.md |
| `good_catch` >= 5 across window | **PROPAGATE**: Document successful pattern in evaluator-tuning.md |

### Accuracy Log Format

Separate from CD events. Appended to session log:

```markdown
### Evaluator Accuracy Events

| Session | Event | Detail | QA Prompt Gap |
|---------|-------|--------|---------------|
| current | false_positive | QA flagged unused import that was actually used in test | Over-aggressive dead code rule |
| current | missed_stub | File upload button rendered but no handler | Runtime verification not performed |
| prev-session | false_negative | Auth bypass via token reuse not caught | No token lifecycle check in checklist |
| current | good_catch | Caught N+1 in nested serializer | SQL logging pattern worked |
```

---

## Cost & Token Tracking

Precise token counts are unavailable on most platforms.
Use proxy metrics that are always measurable.

### Per-Agent Session Log

```markdown
### Resource Usage

| Agent | Turns | Wall-Clock (min) | Sprint Resets | Retries |
|-------|-------|-------------------|---------------|---------|
| pm | 6 | 2.1 | 0 | 0 |
| backend | 18 | 8.4 | 1 | 0 |
| frontend | 15 | 6.7 | 0 | 0 |
| qa | 10 | 4.2 | 0 | 0 |
| **Total** | **49** | **21.4** | **1** | **0** |
```

### How to Record

- **Turn count**: Always available (count progress file updates)
- **Wall-clock time**: Bash timestamps at spawn and completion
- **Sprint resets**: Count checkpoint files per agent
- **Precise tokens**: Available only via `oma stats` post-hoc (parses CLI logs when supported)

### Usage

- Compare turns/time across sessions for similar tasks → detect efficiency regression
- Agents using 2x+ average turns → candidate for prompt or skill refinement
- Track cost delta when scaffold changes are made (supports future harness audits)
