# Reasoning Templates

Use these templates by filling in the blanks when multi-step reasoning is needed.
Complete **each step before moving to the next** to avoid losing direction.

---

## 1. Debugging Reasoning (Debug Agent, Backend/Frontend/Mobile Agent)

Repeat the loop below when finding the cause of a bug. After 3 iterations without resolution, record `Status: blocked`.

```
=== Hypothesis #{N} ===

Observation: {error message, symptoms, reproduction conditions}
Hypothesis: "{phenomenon} is caused by {suspected cause}"
Verification method: {how to verify; code reading, logs, tests, etc.}
Verification result: {what was actually confirmed}
Verdict: Correct / Incorrect

If correct → Move to fix step
If incorrect → Write new hypothesis #{N+1}
```

**Example:**
```
=== Hypothesis #1 ===
Observation: "Cannot read property 'map' of undefined" in TodoList
Hypothesis: "todos is undefined when .map() is called before API response"
Verification method: Check initial value of todos in TodoList component
Verification result: No initial value in useState() → undefined
Verdict: Correct → Set default value of todos to []
```

---

## 2. Architecture Decision (PM Agent, Backend Agent)

Fill in this matrix when technology selection or design decisions are needed.

```
=== Decision: {what needs to be chosen} ===

Options:
  A: {option A}
  B: {option B}
  C: {option C} (if applicable)

Evaluation criteria and scores (1-5):
| Criterion           | A | B | C | Weight |
|---------------------|---|---|---|--------|
| Performance         |   |   |   | {H/M/L} |
| Implementation complexity |   |   |   | {H/M/L} |
| Team familiarity    |   |   |   | {H/M/L} |
| Scalability         |   |   |   | {H/M/L} |
| Existing code consistency |   |   |   | {H/M/L} |

Conclusion: {selected option}
Reason: {1-2 line rationale}
Trade-off: {why giving up advantages of unchosen options}
```

**Example:**
```
=== Decision: State management library ===

Options:
  A: Zustand
  B: Redux Toolkit
  C: React Context

| Criterion           | A | B | C | Weight |
|---------------------|---|---|---|--------|
| Performance         | 4 | 4 | 3 | M     |
| Implementation complexity | 5 | 3 | 4 | H     |
| Team familiarity    | 3 | 5 | 5 | M     |
| Scalability         | 4 | 5 | 2 | M     |
| Existing code consistency | 2 | 5 | 3 | H |

Conclusion: Redux Toolkit
Reason: Existing codebase uses RTK, highest team familiarity
Trade-off: Giving up Zustand's simplicity for consistency
```

---

## 3. Cause-Effect Chain (Debug Agent)

Use this to trace execution flow step-by-step in complex bugs.

```
=== Execution Flow Trace ===

1. [Entry point]   {file:function} - {input value}
2. [Call]          {file:function} - {passed value}
3. [Processing]    {file:function} - {transformation/logic}
4. [Failure point] {file:function} - {unexpected behavior here}
   - Expected: {expected behavior}
   - Actual: {actual behavior}
   - Cause: {why different}
5. [Result]        {error message or incorrect output}
```

**Example:**
```
1. [Entry point]   pages/todos.tsx:TodoPage - user accesses /todos
2. [Call]          hooks/useTodos.ts:useTodos - fetchTodos() called
3. [Processing]    api/todos.ts:fetchTodos - GET /api/todos request
4. [Failure point] hooks/useTodos.ts:23 - data returned as undefined
   - Expected: data = [] (empty array)
   - Actual: data = undefined (before fetch completes)
   - Cause: useQuery initialData not set
5. [Result]        undefined.map() in TodoList → TypeError
```

---

## 4. Refactoring Judgment (All Implementation Agents)

Use this to decide "fix it or leave it as-is" when modifying code.

```
=== Refactoring Judgment ===

Current code issue: {what is the problem}
Relation to task: Directly related / Indirectly related / Unrelated

Directly related → Fix it
Indirectly related → Record in result, fix only within current task scope
Unrelated → Record in result only (never fix)
```

---

## 5. Performance Bottleneck Analysis (Debug Agent, QA Agent)

Systematically find bottlenecks for "it's slow" reports.

```
=== Performance Bottleneck Analysis ===

Measurements:
  - Total response time: {ms}
  - DB query time: {ms} ({N} queries)
  - Business logic: {ms}
  - Serialization/rendering: {ms}

Bottleneck location: {step taking the most time}
Cause: {N+1 query / heavy computation / large response / missing index / ...}
Solution: {specific fix method}
Expected improvement: {X}ms → {Y}ms
```

---

## 6. Exploration Decision (Orchestrator, All Agents)

Use this when the same issue causes repeated failures and alternative approaches should be explored.
See `exploration-loop.md` for the full protocol.

```
=== Exploration Decision ===

Problem: {what needs to be solved}
Current Score: {composite quality score, or N/A if unmeasured}
Attempts So Far: {count and outcomes}

Hypothesis A: {approach description}
  Predicted Impact: {which score dimensions improve/regress}
  Confidence: HIGH / MEDIUM / LOW
  Scope: {files to modify, max 3}

Hypothesis B: {approach description}
  Predicted Impact: {which score dimensions improve/regress}
  Confidence: HIGH / MEDIUM / LOW
  Scope: {files to modify, max 3}

Selection: Highest composite score with confidence >= MEDIUM
Fallback: If all LOW confidence, escalate to user
```

**Example:**
```
=== Exploration Decision ===

Problem: Input validation keeps failing security review
Current Score: 68
Attempts So Far: 2 (both rejected by QA for bypass vulnerability)

Hypothesis A: Zod schema validation at router level
  Predicted Impact: Security +15, Correctness +5
  Confidence: HIGH
  Scope: routes/todo.ts, schemas/todo.ts

Hypothesis B: Custom middleware with sanitization
  Predicted Impact: Security +10, Performance -5
  Confidence: MEDIUM
  Scope: middleware/validate.ts, routes/todo.ts, middleware/index.ts

Selection: Hypothesis A (higher confidence, better predicted impact)
```

---

## Usage Rules

1. **When to use**: Required for Complex difficulty tasks, recommended for Medium
2. **Where to record**: Record reasoning process in `progress-{agent-id}.md`
3. **If blanks cannot be filled**: Gather that information first (Serena, code reading, log checking)
4. **Unresolved after 3 iterations**: `Status: blocked` + include reasoning so far in result
