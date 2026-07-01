# PM Agent - Error Recovery Playbook

When you encounter a failure during planning, follow these recovery steps.

---

## Requirements Ambiguous

**Symptoms**: User request is vague ("Make me a good app", "Make it better")

1. Break down what you DO understand
2. List specific assumptions you're making
3. Create plan based on reasonable assumptions
4. Mark assumptions clearly: `Assumption: [description]`
5. **NEVER do this**: Create tasks while still ambiguous (agents will lose direction)

---

## Existing Codebase Unknown

**Symptoms**: Planning for an existing project but don't know the architecture

1. Use Serena: `get_symbols_overview("src/")` or `get_symbols_overview("app/")`
2. Look for framework indicators: `package.json`, `pyproject.toml`, `pubspec.yaml`
3. Check for existing patterns: `search_for_pattern("@app.get|@app.post")` (FastAPI)
4. If Serena unavailable: note in plan "architecture assumptions: verify before execution"

---

## Task Decomposition Too Granular or Too Coarse

**Self-check**:
- Each task should take 1 agent, 10-20 turns
- If a task needs < 5 turns: merge with a related task
- If a task needs > 30 turns: split into sub-tasks
- If unsure: err on the side of fewer, larger tasks

---

## Dependency Deadlock

**Symptoms**: Task A depends on B, B depends on A (circular)

1. Identify the cycle
2. Break it by defining an API contract or shared interface first
3. Create a priority-0 task: "Define API contracts" (no dependencies)
4. Both tasks then depend on the contract, not on each other

---

## Tech Stack Decision Unclear

**Symptoms**: Multiple valid options, no clear winner

1. Check existing codebase first; consistency wins over "better" tech
2. If greenfield: use the project's default stack (see SKILL.md tech-stack references)
3. Default choices: refer to each domain skill's `resources/tech-stack.md` for current versions
   - Frontend: `oma-frontend/resources/tech-stack.md`
   - Backend: `oma-backend/resources/tech-stack.md`
   - Mobile: `oma-mobile/resources/tech-stack.md`
4. Note decision rationale in plan: `tech_decision: { choice: "X", reason: "Y" }`

---

## Serena Memory / Quota Issues

Same as backend-agent playbook: See relevant sections.

---

## General Principles

- **Plans are not code**: They don't need to be perfect. Agents can adjust during execution
- **Blocked**: If no progress after 5 turns, save current state, `Status: blocked`
- **No code writing**: PM only plans; delegate implementation to other agents
