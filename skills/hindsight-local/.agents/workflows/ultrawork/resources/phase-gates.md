# Phase Gate Definitions

Each phase must pass its gate before proceeding to the next.

---

## PLAN_GATE

**Owner**: PM Agent
**Trigger**: After Steps 1-4

### Criteria
- [ ] Plan documented with acceptance criteria
- [ ] Assumptions explicitly listed
- [ ] Alternatives considered for architecture decisions (min 2)
- [ ] Over-engineering review completed
- [ ] User confirmation received

### Auto-pass Conditions
- Difficulty: Simple
- Existing pattern match
- User explicitly skips

### Failure Action
Revise plan, do not proceed to IMPL

---

## IMPL_GATE

**Owner**: Implementation Agent
**Trigger**: After Step 5

### Criteria
- [ ] Code compiles/builds successfully
- [ ] Tests pass
- [ ] Only planned files modified
- [ ] No unrequested features added
- [ ] Diff reviewed for scope creep

### Auto-pass Conditions
- All tests green
- Diff < 200 lines
- No new dependencies

### Failure Action
Fix issues, re-run implementation

---

## VERIFY_GATE

**Owner**: QA Agent
**Trigger**: After Steps 6-8

### Criteria
- [ ] Implementation matches requirements
- [ ] Zero CRITICAL issues
- [ ] Zero HIGH issues
- [ ] Improvements validated (no regressions)

### Blockers
- Any CRITICAL or HIGH issue

### Failure Action
Return to IMPL with findings

---

## REFINE_GATE

**Owner**: Implementation + Debug Agents
**Trigger**: After Steps 9-13

### Criteria
- [ ] No files > 500 lines (or justified)
- [ ] No functions > 50 lines (or justified)
- [ ] Integration opportunities captured
- [ ] Side effects verified
- [ ] Unused code cleaned

### Skip Conditions
- Simple tasks < 50 lines total change
- User explicitly skips

### Failure Action
Address issues, re-verify

---

## SHIP_GATE

**Owner**: QA Agent
**Trigger**: After Steps 14-17

### Criteria
- [ ] Lint passes
- [ ] Type check passes
- [ ] Test coverage >= 80%
- [ ] UX flows verified
- [ ] No hardcoded secrets
- [ ] Migrations safe
- [ ] Related issues addressed
- [ ] Deployment checklist complete

### Final Approval
User must confirm

### Quality Score Requirement (when measurement is available)
- [ ] Final composite score >= 75 (Grade B or above)
- [ ] Score delta from IMPL baseline >= 0 (no regression)

### Failure Action
Return to appropriate phase based on failure type

---

## Quality Score Integration

Gates from IMPL through SHIP incorporate the Quality Score when measurement is available (see `quality-score.md`).
Quality Score is loaded **conditionally** per `context-loading.md`, not at Phase 0.

When a score is available, it supplements the checklist:
- **Grade A (90-100)**: Gate auto-passes if all checklist items are also met
- **Grade B (75-89)**: Gate passes with noted improvements for next phase
- **Grade C (60-74)**: Gate FAILS; must improve score before proceeding
- **Grade D (0-59)**: Hard FAIL; rollback required

When no measurement tools are available, gates fall back to the binary checklist above.

### Repeated Gate Failure Rule

If the same gate **fails twice** on the same issue:
- Load `exploration-loop.md` (conditional loading, see `context-loading.md`)
- Activate the **Exploration Loop** (see `exploration-loop.md`)
- Generate 2-3 alternative hypotheses
- Experiment and select the highest-scoring approach
- Resume gate evaluation with the winning approach
