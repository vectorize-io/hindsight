# Debug Agent - Execution Protocol

## Step 0: Prepare
1. **Assess difficulty**: see `../../_shared/core/difficulty-guide.md`
   - **Simple**: Skip to Step 3 | **Medium**: All 4 steps | **Complex**: All steps + checkpoints
2. **Check lessons**: read your domain section in `../../_shared/core/lessons-learned.md`
3. **Clarify requirements**: follow `../../_shared/core/clarification-protocol.md`
   - Check **Uncertainty Triggers**: security/auth related bugs, existing code conflict potential?
   - Determine level: LOW → proceed | MEDIUM → present options | HIGH → ask immediately
4. **Use reasoning templates**: for Complex bugs, use `../../_shared/core/reasoning-templates.md` (hypothesis loop, execution trace)
5. **Budget context**: follow `../../_shared/core/context-budget.md` (use find_symbol, not read_file)

**Intelligent Escalation**: When uncertain, escalate early. Don't blindly proceed.

Follow these steps in order (adjust depth by difficulty).

## Step 1: Understand
- Gather: What happened? What was expected? Error messages? Steps to reproduce?
- Read relevant code using Serena:
  - `find_symbol("functionName")`: Locate the failing function
  - `find_referencing_symbols("Component")`: Find all callers
  - `search_for_pattern("error pattern")`: Find similar issues
- Classify: logic bug, runtime error, performance issue, security flaw, or integration failure

## Step 2: Reproduce & Diagnose
- Trace execution flow from entry point to failure
- Identify the exact line and condition that causes the bug
- Determine root cause (not just symptom):
  - Null/undefined access?
  - Race condition?
  - Missing validation?
  - Wrong assumption about data shape?
- Check `resources/common-patterns.md` for known patterns

## Step 3: Fix & Test
- Apply minimal fix that addresses the root cause
- Write a regression test that:
  - Fails without the fix
  - Passes with the fix
  - Covers the specific edge case
- Check for similar patterns elsewhere: `search_for_pattern("same_bug_pattern")`
- If found, fix proactively or report them

## Step 4: Document & Verify
- Run `resources/checklist.md` items
- Save bug report to `.agents/results/bugs/` using `resources/bug-report-template.md`
- Include: root cause, fix, prevention advice
- Verify no regressions in related functionality

## On Error
See `resources/error-playbook.md` for recovery steps.
