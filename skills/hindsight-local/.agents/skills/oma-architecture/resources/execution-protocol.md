# Architecture Agent - Execution Protocol

## Step 0: Prepare
1. Assess difficulty using `../../_shared/core/difficulty-guide.md`
2. Clarify the decision:
   - What is being decided?
   - What constraints already exist?
   - What would make this decision successful?
3. Identify scope:
   - single component/module
   - subsystem
   - cross-cutting system architecture
4. Choose the lightest fitting methodology via `methodology-selection.md`

## Step 1: Frame the Problem
- Separate symptoms from decisions
- Name the architecture concern explicitly:
  - boundary / ownership
  - API shape / caller burden
  - reliability / consistency / scaling
  - migration / investment prioritization
- Record constraints, quality attributes, and non-goals

## Step 2: Gather Context
- Analyze only the code and docs relevant to the decision
- Map existing architecture:
  - key modules or services
  - ownership boundaries
  - integration points
  - current pain points
- If context is vague, start in Diagnostic Mode before deeper analysis

## Step 3: Decide Consultation Depth
- **Simple**: no stakeholder consultation, analyze inline
- **Medium**: consult 1-3 stakeholder agents via `stakeholder-synthesis.md`
- **Complex**: perform a structured stakeholder sweep, then synthesize

## Step 4: Run the Selected Method

### Diagnostic Mode
- Convert vague pain into a concrete architecture problem
- Route into Recommendation, Design-Twice, ATAM-style, or CBAM-style mode

### Recommendation Mode
- Define 2-3 options when the decision is material
- Compare on:
  - boundary clarity
  - quality attributes
  - implementation cost
  - operational cost
  - future change cost

### Design-Twice Mode
- Force at least two materially different options
- Avoid superficial variations on the same decomposition
- Compare and synthesize if needed

### ATAM-style Mode
- Identify quality attribute scenarios
- Surface sensitivity points, tradeoff points, risks, and non-risks
- Prioritize architectural concerns by impact

### CBAM-style Mode
- Compare candidate investments
- Estimate benefit, cost, and sequencing value
- Recommend a prioritized investment path

### ADR Mode
- Produce a concise decision artifact after analysis

## Step 5: Synthesize
- Summarize stakeholder perspectives
- Separate:
  - agreements
  - tensions
  - assumptions
- Make an explicit recommendation
- If a decision is still user-owned, frame the options and tradeoffs clearly

## Step 6: Verify
- Run `checklist.md`
- Confirm the recommendation is:
  - method-appropriate
  - cost-aware
  - scoped correctly
  - explicit about risks and assumptions

## Step 7: Document
- Save artifact to `.agents/results/architecture/`
- Recommended filename patterns:
  - `architecture-recommendation-<topic>.md`
  - `architecture-review-<topic>.md`
  - `adr-<topic>.md`
  - `cbam-<topic>.md`

## Escalation
- If the question is really about task sequencing -> hand off to oma-pm
- If the decision is really infra implementation -> hand off to oma-tf-infra
- If the issue is primarily code correctness -> hand off to oma-debug
- If the issue is primarily security/performance/accessibility verification -> hand off to oma-qa
