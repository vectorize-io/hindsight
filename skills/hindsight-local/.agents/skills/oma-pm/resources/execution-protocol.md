# PM Agent - Execution Protocol

## Step 0: Prepare
1. **Assess difficulty**: see `../../_shared/core/difficulty-guide.md`
   - **Simple**: Lightweight plan, 3-5 tasks | **Medium**: Full 4 steps | **Complex**: Full + API contracts
2. **Clarify requirements**: follow `../../_shared/core/clarification-protocol.md` (critical for PM)
   - Check **Uncertainty Triggers**: business logic, security/auth, existing code conflicts?
   - Determine level: LOW → proceed | MEDIUM → present options | HIGH → ask immediately
3. **Use reasoning templates**: for architecture decisions, use `../../_shared/core/reasoning-templates.md` (decision matrix)
4. **Check lessons**: read cross-domain section in `../../_shared/core/lessons-learned.md`
5. **If governance, risk, or formal planning matters**: read `resources/iso-planning.md`

**Intelligent Escalation**: When uncertain, escalate early. Don't blindly proceed.

Follow these steps in order (adjust depth by difficulty).

## Step 1: Analyze Requirements
- Parse user request into concrete requirements
- Identify explicit and implicit features
- List edge cases and assumptions
- Ask clarifying questions if ambiguous
- Use Serena (if existing codebase): `get_symbols_overview` to understand current architecture
- If risk or governance matters, identify:
  - stakeholders
  - constraints
  - decision owners
  - major delivery risks

## Step 2: Design Architecture
- Select tech stack (frontend, backend, mobile, database, infra)
- Define API contracts (method, path, request/response schema)
- Design data models (tables, relationships, indexes)
- Identify security requirements (auth, validation, encryption)
- Plan infrastructure (hosting, caching, CDN, monitoring)
- When relevant:
  - map plan structure to ISO 21500-style project management concepts
  - record top risks and treatments using ISO 31000-style thinking
  - note governance, responsibility, and approval needs using ISO 38500-style thinking

## Step 3: Decompose Tasks
- Break into tasks completable by a single agent
- Each task has: agent, title, description, acceptance criteria, priority, dependencies, **scope**
- `scope`: array of directory prefixes this agent is allowed to modify (e.g., `["src/api/", "migrations/"]`). Used by `verify` to detect cross-agent boundary violations in parallel execution.
- Minimize dependencies for maximum parallel execution
- Priority tiers: 1 = independent (run first), 2 = depends on tier 1, etc.
- Complexity: Low / Medium / High / Very High
- Save to `.agents/results/plan-{sessionId}.json` and `.agents/results/result-pm.md`

## Step 4: Validate Plan
- Check: Can each task be done independently given its dependencies?
- Check: Are acceptance criteria measurable and testable?
- Check: Is security considered from the start (not deferred)?
- Check: Are API contracts defined before frontend/mobile tasks?
- Check: Are major risks, owners, and approval points explicit when needed?
- Output task-board.md format for orchestrator compatibility

## On Error
See `resources/error-playbook.md` for recovery steps.
