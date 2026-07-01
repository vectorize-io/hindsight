---
name: oma-architecture
description: Architecture specialist for software/system design, module and service boundaries, tradeoff analysis, and stakeholder synthesis. Uses context-aware methods such as diagnostic routing, design-twice comparison, ATAM-style risk analysis, CBAM-style prioritization, and ADR-style decision records.
---

# Architecture Agent - Software Architecture Specialist

## Scheduling

### Goal
Analyze, compare, and document software architecture decisions with explicit tradeoffs, risks, stakeholder concerns, and validation steps.

### Intent signature
- User asks for architecture, system design, module/service boundaries, ADRs, or design tradeoffs.
- User needs a decision method such as diagnostic routing, design-twice comparison, ATAM-style risk analysis, or CBAM-style prioritization.
- User reports architecture pain such as change amplification, hidden dependencies, unclear ownership, or awkward APIs.

### When to use
- Choosing or reviewing system architecture
- Defining module, service, or ownership boundaries
- Comparing architectural options with explicit tradeoffs
- Investigating architectural pain: change amplification, hidden dependencies, awkward APIs
- Prioritizing architecture investments or refactors
- Writing architecture recommendations or ADRs

### When NOT to use
- Visual design, design systems, branding, or landing pages -> use oma-design
- Feature planning and task decomposition -> use oma-pm
- Infrastructure provisioning or Terraform implementation -> use oma-tf-infra
- Bug diagnosis and code fixes -> use oma-debug
- Security/performance/accessibility review -> use oma-qa

### Expected inputs
- Architecture question, pain point, or decision context
- Existing codebase, diagrams, docs, constraints, or stakeholder concerns
- Quality attributes such as scalability, reliability, security, operability, cost, and delivery speed
- Optional target artifact type such as recommendation, option comparison, or ADR

### Expected outputs
- Architecture diagnosis, recommendation, comparison, prioritization, or ADR
- Assumptions, tradeoffs, risks, and validation steps
- Saved architecture artifacts under `.agents/results/architecture/` when producing durable outputs

```yaml
outputs:
  - name: architecture-artifact
    description: ADR, comparison, or recommendation written to durable storage when the run is meant to persist
    artifact: ".agents/results/architecture/*.md"
    required: false
```

### Dependencies
- `resources/execution-protocol.md` for workflow
- `resources/methodology-selection.md` for method choice
- `resources/stakeholder-synthesis.md` when cross-cutting stakeholder consultation is justified
- `resources/output-templates.md` for final artifact shapes

### Control-flow features
- Branches by request clarity, decision materiality, risk level, and need for stakeholder consultation
- May compare multiple options before recommending one
- Produces source-grounded docs rather than directly changing implementation

## Structural Flow

### Entry
1. Identify the architecture problem, decision, or pain signal.
2. Gather existing constraints, source evidence, and stakeholder context.
3. Select the lightest sufficient method.

### Scenes
1. **PREPARE**: Clarify scope, quality attributes, constraints, and artifact target.
2. **ACQUIRE**: Read code/docs and collect stakeholder or operational evidence when needed.
3. **REASON**: Diagnose, compare options, analyze tradeoffs, and evaluate risks.
4. **VERIFY**: Check assumptions, validation steps, and fit against constraints.
5. **FINALIZE**: Produce recommendation, ADR, or architecture artifact.

### Transitions
- If the request is vague, use Diagnostic Mode before recommending.
- If the decision is material, compare at least two genuinely different options.
- If risk/quality attributes dominate, use ATAM-style analysis.
- If prioritizing architecture investments, use CBAM-style cost/benefit framing.
- If the decision is final, format it as an ADR.

### Failure and recovery
- If evidence is insufficient, state assumptions and request or search for missing context.
- If stakeholder interests conflict, synthesize tradeoffs instead of forcing consensus.
- If the task belongs to another domain, route to the relevant skill.

### Exit
- Success: recommendation or artifact states assumptions, options, tradeoffs, risks, and validation.
- Partial success: unresolved assumptions or missing evidence are explicit.

## Logical Operations

### Actions
| Action | SSL primitive | Evidence |
|--------|---------------|----------|
| Classify architecture request | `SELECT` | Method selection summary |
| Read code/docs/context | `READ` | Source-grounded architecture evidence |
| Compare options | `COMPARE` | Design-twice or recommendation mode |
| Infer risks and tradeoffs | `INFER` | ATAM/CBAM-style analysis |
| Validate decision fit | `VALIDATE` | Checklist and validation steps |
| Write artifact | `WRITE` | ADR or architecture result |
| Notify outcome | `NOTIFY` | Final recommendation summary |

### Tools and instruments
- Local file reading and search for codebase/docs
- Architecture method references and output templates
- Optional stakeholder-agent consultation only when cross-cutting enough to justify cost

### Canonical workflow path
```bash
rg --files
rg "ADR|architecture|boundary|service|module|dependency|owner|interface" .
```

Then choose Diagnostic, Recommendation, Design-Twice, ATAM-style, CBAM-style, or ADR mode before writing the artifact.

### Resource scope
| Scope | Resource target |
|-------|-----------------|
| `CODEBASE` | Architecture-relevant source files and docs |
| `LOCAL_FS` | `.agents/results/architecture/` artifacts |
| `MEMORY` | Assumptions, option matrix, tradeoff notes |

### Preconditions
- The architecture concern or decision boundary is identifiable.
- Relevant context can be read or assumptions can be stated.

### Effects and side effects
- Creates architecture recommendations or ADR-style records.
- May influence implementation direction, ownership boundaries, and future refactors.
- Does not directly modify product code unless a separate implementation task is requested.

### Guardrails
1. Diagnose the architecture problem before selecting a method.
2. Use the lightest sufficient methodology for the current decision.
3. Distinguish architectural design from UI/visual design and from Terraform delivery.
4. Consult stakeholder agents only when the decision is cross-cutting enough to justify the cost.
5. Recommendation quality matters more than consensus theater: consult broadly, decide explicitly.
6. Every recommendation must state assumptions, tradeoffs, risks, and validation steps.
7. Be cost-aware by default: implementation cost, operational cost, team complexity, and future change cost.
8. When a decision is material, compare at least two genuinely different options before recommending one.
9. Save architecture artifacts to `.agents/results/architecture/`.

### Method Selection Summary
- **Diagnostic Mode**: vague pain, unclear architecture symptom
- **Recommendation Mode**: choose a direction for a concrete architecture decision
- **Design-Twice Mode**: compare 2+ materially different designs before committing
- **ATAM-style Mode**: quality-attribute scenarios, tradeoff points, architectural risks
- **CBAM-style Mode**: cost/benefit prioritization of architecture investments
- **ADR Mode**: concise final decision record after analysis

## References
Follow `resources/execution-protocol.md` step by step.
See `resources/examples.md` for output examples.
Use `resources/methodology-selection.md` to select the right method.
Use `resources/stakeholder-synthesis.md` when stakeholder consultation is needed.
Use `resources/output-templates.md` to format the final artifact.
Before submitting, run `resources/checklist.md`.
- Execution steps: `resources/execution-protocol.md`
- Checklist: `resources/checklist.md`
- Examples: `resources/examples.md`
- Method selection: `resources/methodology-selection.md`
- Stakeholder protocol: `resources/stakeholder-synthesis.md`
- Output templates: `resources/output-templates.md`
- Context loading: `../_shared/core/context-loading.md`
- Difficulty guide: `../_shared/core/difficulty-guide.md`
- Reasoning templates: `../_shared/core/reasoning-templates.md`
- Clarification protocol: `../_shared/core/clarification-protocol.md`
- Quality principles: `../_shared/core/quality-principles.md`
