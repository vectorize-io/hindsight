# SSL-lite Skill Template

Use this template for OMA skills. Replace placeholders; do not keep placeholder text in the final skill.

````markdown
---
name: oma-{skill-name}
description: >
  {One concise routing description. Include the task/domain and the phrases that should trigger this skill.}
---

# {Skill Title}

## Scheduling

### Goal
{What capability this skill adds and the outcome it should produce.}

### Intent signature
- {Prompt pattern, domain term, or situation that should activate this skill}
- {Another trigger}

### When to use
- {Positive use case}
- {Positive use case}

### When NOT to use
- {Boundary case} -> use {other skill/tool}
- {Boundary case} -> out of scope

### Expected inputs
- `{input_name}`: {meaning}
- `{input_name}`: {meaning}

### Expected outputs

Use freeform bullets when the skill has no machine-checkable artifacts:

- {Primary output}
- {Validation/reporting output}

Or declare a structured `outputs:` block when artifacts can be globbed. `oma verify` reads this block via `parseExpectedOutputs` and fails the closure check when any `required: true` artifact is missing after the agent reports completion.

```yaml
outputs:
  - name: plan
    description: PM task breakdown
    artifact: ".agents/results/plan-*.json"
    required: true
  - name: tests
    description: regression tests
    artifact: "**/test_*.py"
    required: false
```

Field rules:
- `name`: short identifier, lowercase
- `description`: human-readable purpose
- `artifact`: glob relative to workspace root (supports `**`)
- `required`: defaults to `false`; only `true` blocks closure

### Dependencies
- {Tools, files, standards, APIs, local resources}

### Control-flow features
- {Branching, loops, tool calls, user clarification points, write behavior}

## Structural Flow

### Entry
1. {First thing the agent must establish}
2. {Second thing}

### Scenes
1. **PREPARE**: {Setup, scope, assumptions}
2. **ACQUIRE**: {Read files, fetch docs, inspect inputs}
3. **REASON**: {Decide strategy}
4. **ACT**: {Make changes, run commands, produce artifact}
5. **VERIFY**: {Check result}
6. **FINALIZE**: {Report outcome}

### Transitions
- If {condition}, {next action}.
- If {condition}, {next action}.

### Failure and recovery
| Failure | Recovery |
|---------|----------|
| {Failure mode} | {Recovery action} |

### Exit
- Success: {observable success condition}
- Partial success: {what must be reported}
- Failure: {what must be reported}

## Logical Operations

### Actions
| Action | SSL primitive | Evidence |
|--------|---------------|----------|
| {Action} | `{READ|SELECT|COMPARE|VALIDATE|INFER|WRITE|UPDATE_STATE|CALL_TOOL|REQUEST|TRANSFER|NOTIFY|TERMINATE}` | {Source/evidence} |

### Tools and instruments
- {Tool, API, script, local command, reference file}

### Canonical command path
```bash
{primary command}
{verification command}
```

Use `### Canonical workflow path` instead when the skill is decision-heavy rather than command-heavy:

1. {Step}
2. {Step}
3. {Step}

### Resource scope
| Scope | Resource target |
|-------|-----------------|
| `CODEBASE` | {Files/modules/configs} |
| `LOCAL_FS` | {Local input/output artifacts} |
| `PROCESS` | {Commands or long-running processes} |

### Preconditions
- {Required condition before acting}

### Effects and side effects
- {Files written, commands run, network calls, credentials touched, state changed}

### Guardrails
1. {Safety or quality rule}
2. {Safety or quality rule}

## References
- {Resource label}: `resources/{file}.md`
- {Shared reference}: `../_shared/core/{file}.md`
````

## Mapping To SSL

| SSL key family | SSL-lite location |
|----------------|-------------------|
| `skill_id`, `skill_name` | Frontmatter `name`, title |
| `skill_goal` | `Scheduling / Goal` |
| `intent_signature`, `tags`, `top_pattern` | Frontmatter `description`, `Scheduling / Intent signature`, `When to use` |
| `expected_inputs`, `expected_outputs` | `Scheduling / Expected inputs`, `Expected outputs` |
| `dependencies`, `control_flow_features` | `Scheduling / Dependencies`, `Control-flow features` |
| `scene_*`, `entry_conditions`, `exit_conditions`, `next_scene_rules` | `Structural Flow` |
| `logic_step_id`, `act_type`, `actor`, `object`, `instrument` | `Logical Operations / Actions`, `Tools and instruments` |
| `input_args`, `output_binding`, `preconditions`, `effects` | `Expected inputs`, `Expected outputs`, `Preconditions`, `Effects and side effects` |
| `resource_scope`, `resource_target` | `Logical Operations / Resource scope` |
| Source-adjacent evidence | `References` |

## Inline vs Resource Rule

Keep inline:
- Routing description and boundaries
- Expected inputs and outputs
- Structural flow and recovery
- Logical actions, resource scope, preconditions, side effects, guardrails
- One canonical command or workflow path

Move to `resources/`:
- Long examples
- Provider-specific variants
- Detailed protocols
- Large checklists
- Reference material the agent should load only when needed
