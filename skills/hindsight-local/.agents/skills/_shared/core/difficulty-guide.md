# Difficulty Assessment & Protocol Branching

All agents assess task difficulty at the start and apply the appropriate protocol depth.

## Difficulty Assessment Criteria

### Simple
- Single file change
- Clear requirements (e.g., "change button color", "add field")
- Repeating existing patterns
- **Expected turns**: 3-5

### Medium
- 2-3 file changes
- Some design decisions needed
- Applying existing patterns to new domains
- **Expected turns**: 8-15

### Complex
- 4+ file changes
- Architecture decisions required
- Introducing new patterns
- Dependencies on other agent outputs
- **Expected turns**: 15-25

---

## Protocol Branching

### Simple → Fast Track
1. ~~Step 1 (Analyze)~~: Skip; proceed directly to implementation
2. **Pre-check**: Confirm whether test files exist for the target module (e.g., `__tests__/`, `*.test.*`)
3. Step 3 (Implement): Implementation
4. Step 4 (Verify): Minimal checklist items:
   - All `Code Quality` items from `common-checklist.md`
   - `Tests actually assert meaningful behavior` (if tests exist or were added)
   - Run existing tests to verify no regressions

### Medium → Standard Protocol
1. Step 1 (Analyze): Brief
2. Step 2 (Plan): Brief
3. Step 3 (Implement): Full
4. Step 4 (Verify): Full

### Complex → Extended Protocol (Sprint-Based)

1. Step 1 (Analyze): Full + explore existing code with Serena
2. Step 2 (Plan): Full + **decompose into 2-4 feature-focused sprints**
   - Each sprint = independently testable deliverable
   - Target: 5-8 turns per sprint
   - Record sprint plan in `progress-{agent-id}.md`
3. **Sprint Loop** (repeat per sprint):
   - Step 3 (Implement): Current sprint's features only
   - Step 3.5 (Sprint Gate):
     - [ ] Sprint deliverable complete
     - [ ] lint/test pass
     - If sprint took 2x expected turns → write checkpoint and inform user
       (see `context-budget.md` Standalone Agent Mode)
   - On gate pass → next sprint
4. Step 4 (Verify): Full + `common-checklist.md`

#### Sprint Decomposition Example

Task: "JWT auth + CRUD API + tests"
- Sprint 1: User model + auth endpoints (register/login)
- Sprint 2: CRUD endpoints + validation
- Sprint 3: Tests + error handling

---

## Difficulty Misjudgment Recovery

- Started as Simple but more complex than expected → Switch to Medium protocol, record in progress
- Started as Medium but architecture decisions needed → Upgrade to Complex
- Started as Complex but actually simple → Just finish quickly (minimal overhead)
