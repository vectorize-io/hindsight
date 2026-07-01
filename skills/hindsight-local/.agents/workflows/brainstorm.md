---
name: brainstorm
description: Design-first ideation workflow that explores user intent, clarifies constraints, proposes approaches, and produces an approved design document before planning
disable-model-invocation: true
---

# MANDATORY RULES: VIOLATION IS FORBIDDEN

- **Response language follows `language` setting in `.agents/oma-config.yaml` if configured.**
- **NEVER skip steps.** Execute from Step 1 in order.
- **Do NOT write any code.** This workflow produces a design document, not implementation.
- **You MUST use MCP tools throughout the workflow.**
  - Use code analysis tools (`get_symbols_overview`, `find_symbol`, `search_for_pattern`) to analyze the existing codebase.
  - Use memory tools (write/edit) to record design results.
  - Memory path: configurable via `memoryConfig.basePath` (default: `.serena/memories`)
  - Tool names: configurable via `memoryConfig.tools` in `.agents/mcp.json`
  - Do NOT use raw file reads or grep as substitutes.

---

> **Vendor note:** This workflow executes inline (no subagent spawning). All vendors use their native code analysis and file tools.

---

## L1 Decision Events

Emit required L1 decisions by calling `oma state:emit` directly, as documented in `.agents/skills/_shared/runtime/event-spec.md`.

---

## Step 1: Explore Project Context

// turbo
Use MCP code analysis tools to understand the current codebase:
- `get_symbols_overview` for project structure and existing architecture.
- `find_symbol` and `search_for_pattern` to identify relevant modules, patterns, and conventions.
- Summarize what exists and what the user's idea would affect.

---

## Step 2: Ask Clarifying Questions

Ask the user clarifying questions **one at a time**. Prefer multiple-choice options when possible.
Key areas to clarify:
- **Intent**: What problem are they solving? Who is the target user?
- **Scope**: Must-have vs nice-to-have features
- **Constraints**: Tech stack, timeline, existing integrations
- **Success criteria**: How will they know it's done?

Do NOT proceed to Step 3 until you have a clear understanding of the user's intent.

---

## Step 3: Propose Approaches

Present **2-3 distinct approaches** to solve the problem:
- For each approach: summary, pros, cons, effort estimate (S/M/L)
- **Label each approach as `tactical` (patch/workaround/quick win) or `structural` (root-cause/proper engineering).**
- Highlight the **recommended approach** with rationale
- Include a brief trade-off comparison matrix

**Engineering-first default:** the recommended approach MUST be `structural` — addressing the root cause with proper engineering. Deadline pressure, effort delta, and "we'll fix it properly later" are NOT valid grounds for recommending tactical. Recommending `tactical` is only allowed when the problem itself is genuinely throwaway scope (e.g., one-line config flip, deprecated module being removed). The tighter the deadline, the more important it is to do it right the first time.

**You MUST get user confirmation on the chosen approach before proceeding to Step 4.**

After the user chooses an option, emit and verify the required option-selection decision:

```bash
oma state:emit "decision.made" '{"subject":"brainstorm.option-selection","decision":"Proceed with the user-selected approach.","rationale":"The user selected one option after comparing alternatives and tradeoffs."}'
oma state:verify --workflow brainstorm --checkpoint option-selection
```

---

## Step 4: Present Design

Present the detailed design **section by section**, getting user feedback at each step:
- Architecture overview (components, data flow)
- Key interfaces and contracts
- Integration points with existing code
- Edge cases and error handling strategy

Each section requires explicit user approval before moving to the next.

---

## Step 5: Blind Review Round

Before saving the design, run an independent critique round to surface suppressed issues.

Groupthink and authority bias hide real gaps. A blind round, where each perspective critiques independently without seeing others' feedback, surfaces issues the consensus round would have buried.

**Procedure:**

1. **Select 4-8 independent reviewer lenses** appropriate to the design domain. Examples:
   - Software skill: backend, frontend, devops, security, QA, CTO, end-user, docs-writer
   - Infra skill: network, system, security, finops, SRE, compliance, CTO
   - Customize to the feature's stakeholder map.

2. **Independent critique**: for each lens, produce 2-3 concrete criticisms of the Step 4 design without reference to other lenses' feedback. Cover missing items in their specialty, overlaps/redundancies, naming issues, implementation risks.

3. **Consolidate and dedupe** into a unique issue list. Classify:
   - **Tier 1**: critical gap, must resolve before save
   - **Tier 2**: enhancement, should resolve or explicitly defer
   - **Tier 3**: nice-to-have, defer to next version

4. **Check for suppressed compromises**: for each prior design decision where a reviewer voted `PARTIAL→PASS`, verify the objection was answered on principle (regulatory, consumer, architectural) rather than overridden by majority. Restore any principled objection that was suppressed.

5. **Resolve Tier 1 issues** by updating Step 4 design with either new sections in existing files, new files, or explicit out-of-scope declarations.

6. **Present resolved design** to the user for final approval before Step 6.

Skip only if the design is trivially small (1-2 files, low stakes). Otherwise mandatory.

---

## Step 6: Save Design Document

// turbo
Save the approved design:
1. Write to `docs/plans/designs/<NNN>-<feature-name>.md` where `<NNN>` is the next zero-padded 3-digit number (`ls docs/plans/designs/ | grep -E '^[0-9]{3}-' | tail -1`). Do not append `-design` to the filename; the folder already encodes type.
2. Use memory write tool to record design summary for future reference.

---

## Step 7: Transition to Planning

Inform the user that the design phase is complete and suggest:
> "Design approved. Run `/plan` to decompose this into actionable tasks."

The design document will be automatically loaded by the planning workflow as context.
