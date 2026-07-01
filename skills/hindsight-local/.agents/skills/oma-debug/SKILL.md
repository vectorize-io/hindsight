---
name: oma-debug
description: Bug diagnosis and fixing specialist - analyzes errors, identifies root causes, provides fixes, and writes regression tests. Use for bug, debug, error, crash, traceback, exception, and regression work.
---

# Debug Agent - Bug Fixing Specialist

## Scheduling

### Goal
Reproduce, diagnose, minimally fix, and regression-test bugs while preserving scope discipline and documenting root cause.

### Intent signature
- User reports a bug, crash, traceback, exception, error message, performance issue, intermittent failure, or regression.
- User needs root cause analysis plus a minimal code fix and regression test.

### When to use
- User reports a bug with error messages
- Something is broken and needs fixing
- Performance issues or slowdowns
- Intermittent failures or race conditions
- Regression bugs

### When NOT to use
- Building new features -> use Frontend/Backend/Mobile agents
- General code review -> use QA Agent

### Expected inputs
- Error message, failing behavior, reproduction steps, logs, test failure, or affected code path
- Existing tests, stack traces, environment notes, and suspected regression boundary

### Expected outputs
- Root cause explanation
- Minimal fix scoped to the failing behavior
- Regression test or explicit reason it cannot be added
- Bug documentation under `.agents/results/bugs/` when appropriate

### Dependencies
- `resources/execution-protocol.md`, examples, checklist, common patterns, and debugging checklist
- Local codebase search and symbol/reference tools
- Project test, lint, typecheck, and runtime commands

### Control-flow features
- Branches by reproduction success, error class, suspected layer, and verification outcome
- Reads logs/code/tests and writes code/tests/docs
- Must search for similar patterns after fixing

## Structural Flow

### Entry
1. Capture the reported symptom and suspected scope.
2. Reproduce or establish the closest reliable failing signal.
3. Identify affected files, tests, and related patterns.

### Scenes
1. **PREPARE**: Gather symptoms, logs, reproduction path, and verification command.
2. **ACQUIRE**: Read failing code, tests, references, and similar patterns.
3. **REASON**: Isolate root cause and reject unsupported hypotheses.
4. **ACT**: Apply minimal fix and regression test.
5. **VERIFY**: Re-run failing and related checks.
6. **FINALIZE**: Document root cause, fix, test, and residual risk.

### Transitions
- If reproduction fails, use logs/tests to establish a weaker but explicit diagnostic signal.
- If the first fix fails verification, return to root-cause analysis.
- If similar patterns exist, inspect and patch only affected cases.
- If the request is actually feature work, route to the relevant implementation skill.

### Failure and recovery
- If environment is missing, document the blocker and provide the closest static diagnosis.
- If no regression test is feasible, explain why and include manual verification.
- If fix scope grows, stop and call out the broader design issue.

### Exit
- Success: bug is fixed, regression coverage exists, and checks pass.
- Partial success: root cause or verification blocker is explicit.

## Logical Operations

### Actions
| Action | SSL primitive | Evidence |
|--------|---------------|----------|
| Reproduce failure | `CALL_TOOL` | Test/runtime/log command |
| Search affected code | `READ` | Code, tests, symbols, references |
| Compare similar patterns | `COMPARE` | Pattern search |
| Infer root cause | `INFER` | Diagnostic reasoning |
| Write minimal fix | `WRITE` | Code patch |
| Write regression test | `WRITE` | Test patch |
| Verify behavior | `VALIDATE` | Tests/checks |
| Report result | `NOTIFY` | Root-cause summary |

### Tools and instruments
- Project test, lint, typecheck, runtime, and logging commands
- Serena MCP symbol/reference/pattern search when available
- Debugging checklist and bug report template

### Canonical workflow path
```bash
rg "<error-message-or-symbol>"
rg --files
```

Then run the smallest reproduction command first, add a regression test, and re-run the failing check plus related tests.

### Resource scope
| Scope | Resource target |
|-------|-----------------|
| `CODEBASE` | Failing source, tests, and related patterns |
| `LOCAL_FS` | Bug reports and result artifacts |
| `PROCESS` | Reproduction and verification commands |
| `MEMORY` | Hypotheses, root-cause notes, verification evidence |

### Preconditions
- A bug signal, symptom, or failing behavior is available.
- Relevant code and verification path can be inspected or the blocker is stated.

### Effects and side effects
- Mutates source/tests only as needed for the fix.
- May create bug documentation under `.agents/results/bugs/`.

### Guardrails
1. Reproduce first, then diagnose - never guess at fixes
2. Identify root cause, not just symptoms
3. Minimal fix: change only what's necessary
4. Every fix gets a regression test
5. Search for similar patterns elsewhere after fixing
6. Document in `.agents/results/bugs/`

### Serena MCP
- `find_symbol("functionName")`: Locate the function
- `find_referencing_symbols("Component")`: Find all usages
- `search_for_pattern("error pattern")`: Find similar issues

## References
Follow `resources/execution-protocol.md` step by step.
See `resources/examples.md` for input/output examples.
Before submitting, run `resources/checklist.md`.
Vendor-specific execution protocols are injected automatically by `oma agent:spawn`.
Source files live under `../_shared/runtime/execution-protocols/{vendor}.md`.
- Execution steps: `resources/execution-protocol.md`
- Code examples: `resources/examples.md`
- Checklist: `resources/checklist.md`
- Error recovery: `resources/error-playbook.md`
- Bug report template: `resources/bug-report-template.md`
- Common patterns: `resources/common-patterns.md`
- Debugging checklist: `resources/debugging-checklist.md`
- Context loading: `../_shared/core/context-loading.md`
- Reasoning templates: `../_shared/core/reasoning-templates.md`
- Context budget: `../_shared/core/context-budget.md`
- Lessons learned: `../_shared/core/lessons-learned.md`
- Observability handoff: `../oma-observability/SKILL.md` §Integrations — traces/logs by `trace_id`, 6-dim forensics
