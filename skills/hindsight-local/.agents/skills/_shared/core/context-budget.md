# Context Budget Management

The context window is finite. Especially with Flash-tier models, unnecessary loading directly degrades performance.
Follow this guide to use context efficiently.

---

## Core Principles

1. **No full file reads**: Read only necessary functions/classes
2. **No duplicate reads**: Do not re-read files already read
3. **Lazy resource loading**: Load resources only when needed
4. **Maintain records**: Note read files and symbols in progress

---

## File Reading Strategy

### When Using Serena MCP (Recommended)

```
Bad: read_file("app/api/todos.py")          ← entire file 500 lines
Good: find_symbol("create_todo")             ← just that function 30 lines
Good: get_symbols_overview("app/api")        ← function list only
Good: find_referencing_symbols("TodoService") ← usage only
```

### When Reading Files Without Serena

```
Bad: Read entire file at once
Good: Check first 50 lines (imports + class definitions) → read additional functions as needed
```

---

## Resource Loading Budget

### Flash-tier Models (128K context)

| Category | Budget | Notes |
|----------|--------|-------|
| SKILL.md | ~800 tokens | Auto-loaded |
| execution-protocol.md | ~500 tokens | Always loaded |
| Task resource 1 | ~500 tokens | Selected by difficulty |
| Task resource 2 | ~500 tokens | Complex only |
| error-playbook.md | ~800 tokens | On error only |
| **Total resource budget** | **~3,100 tokens** | ~2.4% of total |
| **Working budget** | **~125K tokens** | Everything else |

### Pro-tier Models (1M+ context)

| Category | Budget | Notes |
|----------|--------|-------|
| Resource budget | ~5,000 tokens | Can load generously |
| Working budget | ~1M tokens | Large files possible |

Pro has less budget pressure, but unnecessary loading still diverts attention.

---

## Tracking Read Files (Record in Progress)

Agents record read files/symbols when updating progress:

```markdown
## Turn 3 Progress

### Read Files
- app/api/todos.py: create_todo(), update_todo() (find_symbol)
- app/models/todo.py: Todo class (find_symbol)
- app/schemas/todo.py: entire file (short file, 40 lines)

### Not Yet Read
- app/services/todo_service.py (will read next turn)
- tests/test_todos.py (reference after implementation)

### Work Completed
- Added priority field to TodoCreate schema
```

This approach:
- Prevents reading the same file twice
- Clarifies what to do next turn
- Allows Orchestrator to understand agent state

---

## Large File Handling Strategy

### Files Over 500 Lines

1. Use `get_symbols_overview` to understand structure
2. Read only necessary symbols with `find_symbol`
3. Never read the entire file

### Complex Components (React/Flutter)

1. Read only props/state definitions first
2. Read render/build methods only when modification needed
3. Skip style sections unless they are modification targets

### Test Files

1. Read only after implementation is complete (unnecessary before)
2. Check only existing test patterns (first 1-2 test functions)
3. Write remaining tests following the pattern

---

## Context Overflow Symptoms & Responses

| Symptom | Meaning | Response |
|---------|---------|----------|
| Forgetting previously read code | Context window exhausted | Note key info in progress, make re-referenceable |
| Re-reading the same file | Tracking gap | Check "Read Files" list in progress |
| Output suddenly becomes shorter | Output tokens insufficient | Write only essentials, omit extra explanations |
| Ignoring instructions | Forgot SKILL.md content | Re-reference only execution-protocol essentials |

---

## Context Anxiety Detection & Reset Protocol

Long-running agents degrade in quality as context fills up. Rather than passively
responding to symptoms, agents must actively detect and reset.
Detection is the **Orchestrator's responsibility** via external observation.
Individual agents do NOT self-monitor for anxiety; they focus on their task.

### Detection (Orchestrator Only)

The Orchestrator monitors agent progress files and triggers reset when needed.

#### Trigger Conditions

| Condition | Detection Method | Action |
|-----------|-----------------|--------|
| Turn budget exhaustion | Agent consumed >= 80% of `expected_turns` AND acceptance criteria < 50% complete | **Context Reset** |
| Progress stall | No progress file update for 3+ consecutive monitoring cycles | **Context Reset** |
| Shallow output | Result file contains stub markers or TODO placeholders | **Re-spawn with explicit instruction** |

The Orchestrator checks these conditions during PHASE 4 (Monitor) polling.

### Context Reset Procedure

When a trigger fires, the Orchestrator executes:

1. **Checkpoint**: Save agent's current state
   ```
   write_memory("checkpoint-{agent-id}", content)
   ```
   Content (assembled by Orchestrator from progress file):
   - Completed items with file paths
   - Remaining items with acceptance criteria
   - Key decisions made so far

2. **Terminate**: Stop the current agent run

3. **Re-spawn**: Start a fresh agent with the checkpoint as context
   - **Claude Code**: New Agent tool call with checkpoint in prompt
   - **CLI agents**: `oma agent:spawn` with `--checkpoint checkpoint-{agent-id}`

4. **Resume**: New agent reads checkpoint, continues from remaining items only

### Standalone Agent Mode (no Orchestrator)

When an agent runs outside orchestration (e.g., direct `/backend` invocation),
the Sprint Gate in `difficulty-guide.md` serves as the safety net.
At each Sprint Gate, the agent checks:
- [ ] Current sprint deliverable complete
- [ ] lint/test pass
- If sprint took 2x expected turns → write checkpoint and inform user:
  "Sprint exceeded turn budget. Checkpoint saved. Re-invoke to continue."
