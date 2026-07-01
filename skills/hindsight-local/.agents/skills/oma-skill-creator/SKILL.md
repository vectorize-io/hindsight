---
name: oma-skill-creator
description: >
  Create or update OMA skills in the SSL-lite human-readable format. Use when adding a new
  `.agents/skills/{skill-name}/SKILL.md`, converting an existing skill to the standardized
  Scheduling / Structural Flow / Logical Operations / References structure, or validating
  whether a skill has enough routing, execution, resource, and safety detail.
---

# Skill Creator - SSL-lite Skill Authoring Specialist

## Scheduling

### Goal
Create, revise, and validate OMA skills using the SSL-lite Markdown structure derived from Scheduling-Structural-Logical skill representation while keeping the result readable, executable, and maintainable.

### Intent signature
- User asks to create a new OMA skill, agent skill, specialist skill, or `.agents/skills/*/SKILL.md`.
- User asks to convert or normalize a skill into the project skill format.
- User asks whether a skill has enough scheduling, structural, logical, reference, or canonical execution detail.
- User mentions SSL-lite, Scheduling / Structural / Logical, skill format, skill template, or skill creator.

### When to use
- Creating a new skill under `.agents/skills/{name}/SKILL.md`
- Updating an existing OMA skill to the SSL-lite format
- Adding `Canonical command path` or `Canonical workflow path` to an execution-heavy skill
- Auditing whether a skill is usable by routing, execution, validation, and recovery logic
- Deciding whether detailed examples belong inline or in `resources/`

### When NOT to use
- Installing third-party skills into `$CODEX_HOME/skills` -> use Codex's built-in skill installer (external; not an OMA skill)
- Creating a Codex plugin bundle -> use Codex's plugin tooling (external; not an OMA skill)
- Writing a general project plan unrelated to skill authoring -> use PM Agent
- Editing product code, infrastructure, frontend, backend, or mobile implementation directly -> use the matching specialist skill

### Expected inputs
- `skill_name`: target directory name, preferably `oma-{domain}`
- `skill_goal`: capability the skill should add
- `trigger_intents`: user prompts or domains that should activate the skill
- `boundaries`: when the skill should not be used and which skill should handle those cases
- `execution_model`: whether the skill is command-heavy, workflow-heavy, judgment-heavy, or reference-heavy
- Existing files, tools, commands, resources, or standards the skill must preserve

### Expected outputs
- A complete `SKILL.md` using the SSL-lite top-level sections
- Optional `resources/*.md`, `config/*.yaml`, `scripts/*`, or `assets/*` only when progressive disclosure or deterministic execution requires them
- A validation report with structure checks, routing checks, execution checks, and unresolved assumptions

### Dependencies
- Existing OMA skill conventions in `.agents/skills/*/SKILL.md`
- SSL-lite template in `resources/ssl-lite-template.md`
- Validation checklist in `resources/validation-checklist.md`
- Local filesystem and search tools for reading adjacent skills and checking output

### Control-flow features
- Branches by new skill vs existing skill update
- Branches by command-heavy vs workflow-heavy vs reference-heavy skill type
- Reads and writes local skill files
- Must preserve user edits and avoid unrelated rewrites

## Structural Flow

### Entry
1. Identify whether the task is creating a new skill, updating an existing skill, or auditing a skill.
2. Read nearby or analogous OMA skills before choosing wording and structure.
3. Determine the skill type: command-heavy, workflow-heavy, judgment-heavy, or reference-heavy.

### Scenes
1. **PREPARE**: Collect skill purpose, trigger intents, boundaries, inputs, outputs, dependencies, and risk/resource constraints.
2. **ACQUIRE**: Read analogous skills, existing resources, project conventions, and any user-provided source material.
3. **REASON**: Decide what belongs inline in `SKILL.md` and what belongs in `resources/`, `config/`, `scripts/`, or `assets/`.
4. **ACT**: Create or update the skill using the SSL-lite template.
5. **VERIFY**: Run structural, routing, execution, resource, and formatting checks.
6. **FINALIZE**: Report created/changed files, validation result, and any remaining assumptions.

### Transitions
- If the skill performs fragile or repeated commands, include `### Canonical command path` inline.
- If the skill is mostly human judgment or investigation, include `### Canonical workflow path` inline.
- If detailed examples are long, variant-specific, or optional, move them to `resources/` and reference them explicitly.
- If a skill already exists, preserve working content and normalize only what is needed for the target format.
- If the trigger description is too broad, narrow it before writing the skill.

### Failure and recovery
| Failure | Recovery |
|---------|----------|
| Skill scope overlaps heavily with another skill | Add a clear `When NOT to use` boundary and cross-route |
| Execution path is vague | Add canonical command or workflow path inline |
| `SKILL.md` becomes too long | Move detailed examples to `resources/` and keep navigation in `References` |
| No reliable validation command exists | Use structural grep/awk checks and manual checklist validation |
| User input is underspecified | Make conservative assumptions and list them, unless the target behavior would be unsafe |

### Exit
- Success: the skill has the four SSL-lite top-level sections, a canonical path, clear routing boundaries, resource constraints, and validation evidence.
- Partial success: the structure is valid but missing commands, tools, source material, or user decisions are reported.
- Failure: no skill file is written or the blocking ambiguity is explicit.

## Logical Operations

### Actions
| Action | SSL primitive | Evidence |
|--------|---------------|----------|
| Read analogous skills | `READ` | Existing `.agents/skills/*/SKILL.md` |
| Select skill type | `SELECT` | Command/workflow/judgment/reference-heavy classification |
| Infer boundaries | `INFER` | Trigger intents and adjacent skill overlap |
| Write skill file | `WRITE` | New or updated `SKILL.md` |
| Add resources | `WRITE` | `resources/`, `config/`, `scripts/`, or `assets/` |
| Validate structure | `VALIDATE` | Heading and canonical-path checks |
| Report result | `NOTIFY` | Changed files and validation summary |

### Tools and instruments
- `rg`, `find`, `awk`, `sed`, `git diff --check`
- `apply_patch` for manual file edits
- Existing OMA skills as examples
- `resources/ssl-lite-template.md` for the canonical section skeleton
- `resources/validation-checklist.md` for acceptance criteria

### Canonical workflow path
1. Read 1-3 analogous existing skills before writing.
2. Classify the new skill as command-heavy, workflow-heavy, judgment-heavy, or reference-heavy.
3. Draft `SKILL.md` from `resources/ssl-lite-template.md`.
4. Add exactly one inline canonical path:
   - `### Canonical command path` for fragile or repeatable commands
   - `### Canonical workflow path` for decision, review, design, or research flow
5. Move long examples, provider-specific details, or optional protocols into `resources/`.
6. Validate top-level headings, canonical path presence, frontmatter, and whitespace.

### Resource scope
| Scope | Resource target |
|-------|-----------------|
| `CODEBASE` | `.agents/skills/*/SKILL.md`, adjacent resources, project skill conventions |
| `LOCAL_FS` | New skill directories and resource files |
| `PROCESS` | Validation commands and optional generator/check scripts |
| `MEMORY` | User requirements, assumptions, and validation notes |

### Preconditions
- The target skill name and purpose are known or can be inferred safely.
- Adjacent skill overlap has been checked.
- The user has authorized changes under `.agents/skills/` when project rules would otherwise protect `.agents/`.

### Effects and side effects
- Creates or modifies files under `.agents/skills/`.
- May add resource files to support progressive disclosure.
- Does not stage or commit changes unless explicitly requested.

### Guardrails
1. Keep the four top-level sections exactly: `Scheduling`, `Structural Flow`, `Logical Operations`, `References`.
2. Keep YAML frontmatter with clear `name` and `description`; routing depends on description quality. Run `oma skills audit` after editing description to confirm the new wording does not collide with adjacent skills (warn ≥ 60%, fail ≥ 75% TF-IDF cosine).
3. Include concrete `When NOT to use` boundaries and cross-routes to adjacent skills.
4. Include expected inputs, expected outputs, dependencies, and control-flow features. Prefer the structured `outputs:` block when artifacts can be globbed so `oma verify` can perform a closure check.
5. Include structural scenes using SSL scene vocabulary where practical: `PREPARE`, `ACQUIRE`, `REASON`, `ACT`, `VERIFY`, `RECOVER`, `FINALIZE`.
6. Include logical actions with SSL primitives such as `READ`, `SELECT`, `VALIDATE`, `INFER`, `WRITE`, `CALL_TOOL`, `NOTIFY`, and `TERMINATE`.
7. Include resource scope and resource targets for filesystem, codebase, process, credentials, network, user data, or memory.
8. Include effects and side effects, especially for commands, network calls, credentials, destructive actions, generated files, and long-running processes.
9. Add one canonical path inline so an agent can execute or reason without loading every resource file.
10. Put long variant-specific details in `resources/`, not in the main skill body.
11. Do not create extra README, changelog, or installation docs inside a skill.
12. Do not overwrite unrelated user edits.

## References
- SSL-lite template: `resources/ssl-lite-template.md`
- Validation checklist: `resources/validation-checklist.md`
- Shared context loading: `../_shared/core/context-loading.md`
- Shared quality principles: `../_shared/core/quality-principles.md`
- Skill utility eval: when creating a new skill, consider adding held-out task fixtures under `.agents/eval/<skill>/` so `oma skills eval` can measure whether the skill improves task outcomes. See `web/docs/guide/skill-eval.md` for the fixture schema and checker types.
