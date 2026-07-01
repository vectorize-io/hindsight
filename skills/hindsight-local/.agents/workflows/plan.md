---
name: plan
description: PM planning workflow that gathers requirements, decomposes them into prioritized tasks, defines API contracts, and produces both a machine-readable plan and a human-readable tracker in docs/plans/
disable-model-invocation: true
---

# MANDATORY RULES: VIOLATION IS FORBIDDEN

- **Response language follows `language` setting in `.agents/oma-config.yaml` if configured.**
- **NEVER skip steps.** Execute from Step 1 in order.
- **You MUST use MCP tools throughout the workflow.**
  - Use code analysis tools (`get_symbols_overview`, `find_symbol`, `search_for_pattern`) to analyze the existing codebase.
  - Use memory tools (write/edit) to record planning results.
  - Memory path: configurable via `memoryConfig.basePath` (default: `.serena/memories`)
  - Tool names: configurable via `memoryConfig.tools` in `.agents/mcp.json`
  - Do NOT use raw file reads or grep as substitutes.

---

> **Vendor note:** This workflow executes inline (no subagent spawning). All vendors use their native code analysis tools. Plan artifacts (`.agents/results/plan-{sessionId}.json` and `docs/plans/work/{NNN}-{name}.md`) are consumed by `/orchestrate` or `/work`, which handle their own vendor detection.

---

## L1 Decision Events

Emit required L1 decisions by calling `oma state:emit` directly, as documented in `.agents/skills/_shared/runtime/event-spec.md`.

---

## Core Philosophy

**Plans are first-class artifacts**: structured, templated, and consumed by other workflows. They are local working artifacts (not committed to the repo; `docs/plans/` is gitignored), but they follow strict conventions so any agent can read and update them.

Two artifacts per plan:

1. **Machine-readable**: `.agents/results/plan-{sessionId}.json` consumed by `/orchestrate` and `/work`.
2. **Human-readable**: `docs/plans/work/{NNN}-{name}.md` with task table, decision log, and progress notes. Lifecycle is tracked via the `Status` field in the file header (`Active` → `Completed`); no folder moves required.

### Layout

```
docs/plans/
├── designs/                       ← permanent design references (Status: Approved/Draft/Superseded)
│   └── {NNN}-{name}.md
└── work/                          ← execution plans (Status: Active/Completed)
    ├── {NNN}-{name}.md
    └── tech-debt-tracker.md
```

- Folder = type (designs vs work). Status field = lifecycle.
- Filename always uses 3-digit zero-padded sequential prefix (`001-`, `002-`, …) per folder.
- Determine the next number with `ls docs/plans/{designs,work}/ | grep -E '^[0-9]{3}-' | tail -1`.
- Plan content language follows the top-of-file rule (`oma-config.yaml` `language` setting). Mixed-language guidance lives in `.agents/rules/i18n-guide.md`.

---

## Step 1: Gather Requirements

Ask the user to describe what they want to build. Clarify:
- Target users
- Core features (must-have vs nice-to-have)
- Constraints (tech stack, existing codebase)
- Deployment target (web, mobile, both)

---

## Step 2: Analyze Technical Feasibility

// turbo
If an existing codebase exists, use MCP code analysis tools to scan:
- `get_symbols_overview` for project structure and architecture patterns.
- `find_symbol` and `search_for_pattern` to identify reusable code and what needs to be built.

Also search `docs/plans/work/` for related past or in-progress plans, and `docs/plans/designs/` for prior design references. Reuse patterns from similar work.

---

## Step 3: Assess Complexity

Use `.agents/skills/_shared/core/difficulty-guide.md` to classify:

- **Simple** → no plan artifact needed; execute directly via `/work`.
- **Medium** → produce both JSON and a lightweight markdown tracker (skip Step 4 API contracts if not cross-boundary).
- **Complex** → produce both artifacts with all sections plus API contracts.

Report scope assessment to the user. Get confirmation before proceeding.

---

## Step 4: Define API Contracts

// turbo
If the plan involves cross-boundary work (frontend ↔ backend, service ↔ service):

1. Design API contracts using `.agents/skills/_shared/core/api-contracts/template.md` (definition/template only — SSOT). Per endpoint:
   - Method, path, request/response schemas
   - Auth requirements, error responses
2. Save the generated contract to `.agents/results/api-contracts/{contract-name}.md` (run artifact; gitignored). If the contract must be versioned as a durable spec, promote it to `docs/plans/contracts/{contract-name}.md` when committing the feature.
3. Reference from the markdown tracker generated in Step 6.
4. Emit and verify the required API contract decision:
   ```bash
   oma state:emit "decision.made" '{"subject":"plan.api-contract","decision":"Use the approved endpoint and contract shape for this plan.","rationale":"The cross-boundary API contract has been reviewed and accepted before task decomposition."}'
   oma state:verify --workflow plan --checkpoint api-contract
   ```

---

## Step 5: Decompose into Tasks

// turbo
Break down the project into actionable tasks. Each task must have:
- Assigned agent (frontend/backend/mobile/qa/debug)
- Title, acceptance criteria
- Priority (P0–P3), dependencies

**Engineering-first decomposition:** prefer tasks that address root causes over tasks that patch individual symptoms. When a deliberate workaround or hotfix is included, record the reason in the Decision Log.

---

## Step 6: Review Plan with User

Present the full plan: task list, priority tiers, dependency graph, agent assignments, completion criteria.
**You MUST get user confirmation before proceeding to Step 7.**

---

## Step 7: Save Plan Artifacts

// turbo
Generate both artifacts.

### 7a. Machine-readable plan

Save `.agents/results/plan-{sessionId}.json` and write a memory summary via the configured memory tool.

### 7b. Human-readable tracker (Medium/Complex only)

Generate `docs/plans/work/{NNN}-{name}.md` using this template (replace `{NNN}` with the next zero-padded 3-digit number for the `work/` folder):

```markdown
# {Plan Title}

> {One-line goal}

**Status**: Active
**Created**: {date}
**Owner**: {agent or human}

## Goal
{What this plan achieves — clear, testable outcome}

## Context
{Relevant background, related code, prior decisions}

## Constraints
{Rules, dependencies, compatibility requirements}

## Tasks

| # | Task | Agent | Priority | Status | Dependencies |
|---|------|-------|----------|--------|--------------|
| 1 | {task} | {agent} | P0 | TODO | — |
| 2 | {task} | {agent} | P0 | TODO | 1 |
| 3 | {task} | {agent} | P1 | TODO | 1, 2 |

## Done When
{Testable completion criteria}
- [ ] {criterion 1}
- [ ] {criterion 2}

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| {date} | {what was decided} | {why} |

## Progress Notes
{Append-only log of progress updates}

- [{date}] Plan created
```

### Naming Convention

- Format: `{NNN}-{kebab-name}.md` (e.g., `008-add-user-authentication.md`).
- `{NNN}` is the next zero-padded 3-digit sequential number for that folder. Determine it from the existing files: `ls docs/plans/work/ | grep -E '^[0-9]{3}-' | tail -1`.
- `{kebab-name}` describes the feature; do **not** append `-design` or `-plan` (the folder already encodes type).
- Lifecycle is tracked via the `Status` header in the file, not via folder moves.

The plan is now ready for `/work` or `/orchestrate` to execute.

---

## Lifecycle Updates (during execution)

`/orchestrate` and `/work` update the markdown tracker as work progresses:

- Task status: `TODO` → `WIP` → `DONE` or `BLOCKED`
- Append timestamped entries to **Progress Notes**
- Record cross-cutting decisions in the **Decision Log**

When all "Done When" criteria are met:

1. Set the header `Status` field: `Active` → `Completed`.
2. Append a completion summary to Progress Notes with the date.
3. The file stays in `docs/plans/work/`; no move required.
4. If any tech debt was introduced, update `docs/plans/work/tech-debt-tracker.md`.

To list in-progress plans: `grep -l "^\*\*Status\*\*: Active" docs/plans/work/*.md`.

---

## Tech Debt Tracker

`docs/plans/work/tech-debt-tracker.md` tracks known debt across all plans:

```markdown
# Tech Debt Tracker

| # | Debt | Source Plan | Priority | Proposed Resolution |
|---|------|-------------|----------|---------------------|
| 1 | {description} | {plan-name} | P1 | {how to fix} |
```

- Add entries when shortcuts are taken during plan execution.
- Remove entries when debt is resolved.
- Review periodically; debt items can become plans themselves.
