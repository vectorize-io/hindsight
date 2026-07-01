---
name: oma-brainstorm
description: Design-first ideation that explores user intent, constraints, and approaches before any planning or implementation. Use for brainstorming, ideation, exploring concepts, and evaluating approaches.
---

# Brainstorm - Design-First Ideation

## Scheduling

### Goal
Explore user intent, constraints, and alternative approaches before planning or implementation, then preserve an approved design for downstream planning.

### Intent signature
- User says they have an idea, want to brainstorm, compare approaches, explore concepts, or design before planning.
- Request is ambiguous enough that implementation or task planning would be premature.

### When to use
- Exploring a new feature idea before planning
- Understanding user intent and constraints before committing to an approach
- Comparing multiple design approaches with trade-offs
- When the user says "I have an idea" or "let's design something"
- Before invoking `/plan` for complex or ambiguous requests

### When NOT to use
- Requirements are already clear and well-defined -> use `oma-pm` directly
- Implementing actual code -> delegate to specialized agents
- Performing code reviews -> use `oma-qa`
- Debugging existing issues -> use `oma-debug`

### Expected inputs
- Early idea, ambiguous goal, product concept, design question, or set of constraints
- Existing project context when the idea must fit a codebase or product direction
- User preferences and approval gates

### Expected outputs
- Clarified intent and constraints
- Two or three approaches with tradeoffs and a recommended option
- Section-by-section approved design document
- Saved design artifact before handoff to planning

### Dependencies
- Shared context loading, reasoning templates, clarification protocol, quality principles, and skill routing
- Downstream PM workflow for task decomposition after design approval

### Control-flow features
- Branches by ambiguity, user answers, approach comparison, and approval gates
- Asks one question at a time
- Stops before implementation or task planning

## Structural Flow

### Entry
1. Confirm that the request is exploratory rather than ready for implementation.
2. Load enough project context to understand constraints.
3. Start with intent and constraints, not solutions.

### Scenes
1. **PREPARE**: Explore context and frame the design question.
2. **ACQUIRE**: Ask clarifying questions one at a time.
3. **REASON**: Generate two or three approaches with tradeoffs.
4. **VERIFY**: Get user approval section by section.
5. **FINALIZE**: Save design and transition to planning when appropriate.

### Transitions
- If requirements become clear and implementation-ready, transition to PM planning.
- If user rejects an approach, revise before moving to detailed design.
- If implementation pressure appears early, defer it until design approval.

### Failure and recovery
- If the user cannot answer a question, propose assumptions and ask for confirmation.
- If scope expands, split the design into smaller sections.
- If alternatives collapse into one option, identify the real constraint causing that.

### Exit
- Success: approved design exists and is ready for planning.
- Partial success: open questions and assumptions are explicit.

## Logical Operations

### Actions
| Action | SSL primitive | Evidence |
|--------|---------------|----------|
| Read context and idea | `READ` | User prompt and project context |
| Ask targeted questions | `REQUEST` | Clarification phase |
| Compare approaches | `COMPARE` | Tradeoff matrix |
| Infer recommendation | `INFER` | Recommended option |
| Validate approval | `VALIDATE` | Section-by-section confirmation |
| Write design artifact | `WRITE` | `docs/plans/designs/` and memory |
| Transition to plan | `NOTIFY` | Handoff summary |

### Tools and instruments
- Context loading, reasoning templates, clarification protocol
- Project memory and `docs/plans/designs/` for persisted designs

### Canonical workflow path
```text
1. Ask one clarifying question at a time.
2. Present 2-3 approaches with tradeoffs and a recommended option.
3. Save the approved design to `docs/plans/designs/` before handing off to planning.
```

### Resource scope
| Scope | Resource target |
|-------|-----------------|
| `MEMORY` | User intent, assumptions, decisions |
| `CODEBASE` | Existing project context when relevant |
| `LOCAL_FS` | Approved design artifacts |

### Preconditions
- The user is still exploring or the request is ambiguous.
- The agent can ask clarifying questions before implementation.

### Effects and side effects
- Produces design decisions and persisted design docs.
- Influences downstream planning but does not implement code.

### Guardrails
1. **No implementation or planning before design approval** - brainstorm produces a design document, not code or task plans
2. **One question at a time** - ask clarifying questions sequentially, not in batches
3. **Always propose 2-3 approaches** - include a recommended option with trade-off analysis
4. **Section-by-section design** - present design incrementally with user confirmation at each step
5. **YAGNI** - do not over-engineer; design only what is needed for the stated goal
6. **Save design, then transition** - persist the approved design document before handing off to `/plan`

### Execution Phases
Follow the brainstorm workflow step by step:
1. **Phase 1 - Context**: Explore the existing codebase and understand the project landscape
2. **Phase 2 - Questions**: Ask clarifying questions one at a time to understand intent and constraints
3. **Phase 3 - Approaches**: Propose 2-3 approaches with a recommended option and trade-off matrix
4. **Phase 4 - Design**: Present the detailed design section by section, getting user approval at each step
5. **Phase 5 - Documentation**: Save the approved design to `docs/plans/designs/` and project memory
6. **Phase 6 - Transition**: Hand off to `/plan` for task decomposition

### Common Pitfalls
- **Jumping to solutions**: Asking "how" before fully understanding "what" and "why"
- **Too many questions at once**: Overwhelming the user with a wall of questions
- **Single approach bias**: Presenting only one option without alternatives
- **Over-engineering**: Designing for hypothetical future requirements instead of stated needs
- **Skipping confirmation**: Moving forward without explicit user approval on design decisions

## References
Vendor-specific execution protocols are injected automatically by `oma agent:spawn`.
Source files live under `../_shared/runtime/execution-protocols/{vendor}.md`.
- Context loading: `../_shared/core/context-loading.md`
- Reasoning templates: `../_shared/core/reasoning-templates.md`
- Clarification protocol: `../_shared/core/clarification-protocol.md`
- Quality principles: `../_shared/core/quality-principles.md`
- Skill-to-agent mapping: `../_shared/core/skill-routing.md`
