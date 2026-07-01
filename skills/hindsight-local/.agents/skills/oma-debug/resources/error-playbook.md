# Debug Agent - Error Recovery Playbook

When you encounter a failure during debugging, follow these recovery steps.
Do NOT stop or ask for help until you have exhausted the playbook.

---

## Cannot Reproduce the Bug

**Symptoms**: Bug described by user but you can't trigger it

1. Re-read user's reproduction steps; are you following them exactly?
2. Check environment differences: browser, OS, node/python version
3. Check data-dependent: does it need specific DB state or test data?
4. Check timing: is it a race condition? Try adding delays or rapid repetition
5. **After 3 attempts**: Record `Status: cannot_reproduce` in result with condition list
   - NEVER give up immediately saying "cannot reproduce"

---

## Fix Introduces New Failure

**Symptoms**: Original bug fixed but other tests break

1. Read the failing tests; are they testing the old (buggy) behavior?
2. If yes: update tests to reflect correct behavior
3. If no: your fix has side effects. Revert and try a more targeted approach
4. `find_referencing_symbols("fixedFunction")` to check all callers
5. Consider: is the function contract changing? If so, update all callers

---

## Root Cause Unclear

**Symptoms**: You see the failure but can't trace why

1. Add logging at each step of the execution path
2. Binary search: is the bug before or after the midpoint?
3. `search_for_pattern("suspicious_pattern")` to find related code
4. Check git history: `git log --oneline -20 -- path/to/file`. When was it last changed?
5. Check: is it a dependency issue? Library version mismatch?
6. **No progress after 5 turns**: Record current analysis in progress, switch to different hypothesis

---

## Bug Is in Another Agent's Domain

**Symptoms**: Frontend bug caused by backend API, or vice versa

1. Confirm: is the root cause really in the other domain?
2. Document the cross-domain issue clearly:
   - Which endpoint/component is wrong
   - What the correct behavior should be
   - Evidence (request/response logs, stack trace)
3. Record in result: `cross_domain_issue: {agent: "backend", description: "..."}`
4. **Do NOT modify directly**; touching another agent's code causes conflicts

---

## Performance Bug Hard to Measure

**Symptoms**: "It's slow" but no clear metric

1. Establish baseline: measure current response time / render time
2. Backend: enable SQL query logging, count queries, check `EXPLAIN ANALYZE`
3. Frontend: run Lighthouse, check React DevTools Profiler
4. Mobile: use Flutter DevTools performance tab
5. Profile before fixing; never optimize without data

---

## Test Cannot Be Written

**Symptoms**: Bug is real but hard to test (race condition, environment-specific)

1. Try: mock the timing / environment condition
2. Try: integration test instead of unit test
3. If truly untestable: document the manual reproduction steps
4. Add a comment in code explaining why the fix is correct
5. Note in result: `test_limitation: "reason why automated test is not feasible"`

---

## Rate Limit / Quota / Memory Fallback

Same as the backend playbook: see `../../oma-backend/resources/error-playbook.md` §"Rate Limit / Quota Error (Gemini API)" and §"Serena Memory Unavailable".

---

## General Principles

- **After 3 failures**: If same approach fails 3 times, must try a different method
- **Blocked**: If no progress after 5 turns, save current state, `Status: blocked`
- **Out of scope**: Other agent's domain. Only record, do not modify directly
