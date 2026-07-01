# Exploration Loop (Hypothesis-Driven)

Transforms the reactive "fix what's broken" pattern into a proactive "explore alternatives, pick the best."
Inspired by autoresearch's continuous hypothesis → experiment → evaluate → keep/discard loop.

---

## When to Activate

The Exploration Loop activates **only** when reactive fixing has failed:

| Condition | Example |
|-----------|---------|
| **Any gate fails twice** on the same issue | VERIFY_GATE, IMPL_GATE, REFINE_GATE, or SHIP_GATE |
| **Quality Score delta is negative** after a fix attempt | Fixing performance regresses correctness |
| **User explicitly requests** exploration | "Try a few approaches and pick the best" |

Do NOT activate for:
- First-attempt implementations (try the standard approach first)
- Simple bug fixes with obvious solutions
- Trivial changes (formatting, naming)

---

## Exploration Protocol

### Step 1: Hypothesize

Generate 2-3 alternative approaches using the **Exploration Decision** reasoning template (see `reasoning-templates.md` #6):

```
=== Exploration Decision ===
Problem: {what needs to be solved}
Current Score: {quality score}
Attempts So Far: {count and outcomes}

Hypothesis A: ...
Hypothesis B: ...
```

### Step 2: Experiment

Execute each hypothesis **in isolation**.

**In multi-agent mode** (`/orchestrate`, `/work`):
- Spawn the **same agent type** (e.g., `backend-engineer`) multiple times with different prompts
- Each spawn includes the hypothesis context in the task description:
  ```
  Task: "Fix input validation using Hypothesis A: Zod schema at router level.
         Context: Previous attempt (raw regex) failed QA twice."
  ```
- Agents use existing IDs; no new agent definitions needed
- Each agent works in a separate workspace (`-w ./hyp-a`, `-w ./hyp-b`)
- Result files differentiated by workspace, not agent ID

**In single-agent mode** (`/ultrawork` inline):
- Execute sequentially: try A → measure → stash/revert → try B → measure → stash/revert
- Use `git stash` or branch per experiment
- Keep all measurements for comparison

### Step 3: Measure

Score each experiment using Quality Score protocol (load `quality-score.md` if not already loaded):

```markdown
### Exploration Results

| Hypothesis | Composite | Correctness | Security | Performance | Coverage | Consistency |
|-----------|-----------|-------------|----------|-------------|----------|-------------|
| A | 82 | 85 | 90 | 70 | 75 | 90 |
| B | 87 | 90 | 85 | 85 | 80 | 95 |

Winner: Hypothesis B (score: 87, delta from current: +15)
```

### Step 4: Select

```
best = max(hypothesis_scores)

IF best.score > current_score:
    KEEP best → merge from workspace or apply stash
    DISCARD others → clean up workspaces
    Record ALL experiments in Experiment Ledger (kept and discarded)
ELSE:
    KEEP current approach (exploration found no improvement)
    Record as "exploration inconclusive"
    ESCALATE to user for guidance
```

### Step 5: Record

Log all experiments in the Experiment Ledger (see `experiment-ledger.md`), including discarded ones:

```markdown
| # | Phase | Agent | Hypothesis | Score Before | Score After | Delta | Decision |
|---|-------|-------|-----------|-------------|------------|-------|----------|
| 4 | EXPLORE | backend | Zod schema validation | 68 | 82 | +14 | DISCARD (not best) |
| 5 | EXPLORE | backend | Middleware sanitization | 68 | 87 | +19 | KEEP (winner) |
```

---

## Constraints

| Constraint | Value | Rationale |
|-----------|-------|-----------|
| Max hypotheses per round | 3 | Diminishing returns; keeps within turn budget |
| Max exploration rounds per session | 2 | Prevents infinite exploration |
| Max turns per hypothesis experiment | 10 | Scoped to focused changes |
| Min score gap to justify exploration | 5 points | Don't explore if current is close to threshold |

---

## Integration with Workflows

### In `/ultrawork`

Triggered at VERIFY or REFINE phase when the same gate fails twice:

```
VERIFY_GATE fails (2nd time, same issue)
  → Load exploration-loop.md (conditional, per context-loading.md)
  → Generate hypotheses (Exploration Decision template)
  → Experiment sequentially (inline, git stash per attempt)
  → Score each, select winner
  → Resume gate evaluation with winning approach
```

### In `/orchestrate`

Triggered when agent verification fails after max retries:

```
Agent FAIL after 2 retries
  → Load exploration-loop.md
  → Spawn same agent type with different hypothesis prompts (parallel, separate workspaces)
  → Collect results, score each
  → Keep winner workspace, discard others
```

### In `/work`

Triggered when Issue Remediation Loop stalls (same issue persists after fix):

```
Same CRITICAL/HIGH issue persists after fix attempt
  → Load exploration-loop.md
  → Re-spawn agent with alternative hypothesis prompts
  → QA scores each result
  → Best result adopted
```

---

## Integration Points

| Component | How It Uses Exploration Loop |
|-----------|----------------------------|
| **Quality Score** | Provides measurement for hypothesis comparison |
| **Experiment Ledger** | Records all hypotheses (kept and discarded) |
| **Phase Gates** | Repeated gate failure triggers exploration |
| **Reasoning Templates** | #6 Exploration Decision provides structured format |
| **Context Loading** | Loaded conditionally, only when triggered |
| **Memory Protocol** | Uses same memory tools for experiment recording |
