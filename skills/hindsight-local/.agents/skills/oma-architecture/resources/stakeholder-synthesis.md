# Stakeholder Synthesis Protocol

Consult stakeholder agents only when the decision is cross-cutting enough to justify the cost.

## Consultation Thresholds

### Solo Analysis

Use when:
- single module or local subsystem
- no major cross-domain effects
- decision can be made from direct code/context analysis

### Targeted Consultation

Consult 1-3 agents when:
- the decision affects multiple layers
- tradeoffs are real but bounded
- specific specialist input is needed

Typical pairings:
- oma-pm: business constraints, scope, product priorities
- oma-backend: service/API/domain concerns
- oma-db: data ownership, schema, consistency
- oma-tf-infra: deployment and operational architecture
- oma-qa: risk, performance, security, testability
- oma-frontend / oma-mobile: client integration and complexity costs

### Full Stakeholder Sweep

Use when:
- architecture is system-wide
- multiple teams/domains are materially affected
- the decision will constrain future roadmap work

## Synthesis Rules

1. Capture perspectives as inputs, not votes
2. Separate:
   - agreements
   - conflicts
   - assumptions
3. Name the real tradeoff behind disagreements
4. Make an explicit recommendation even if perspectives differ
5. If user decision is required, present framed options rather than an unstructured summary

## Output Section

Include a section like:

```md
## Stakeholder Perspectives
- PM: ...
- Backend: ...
- DB: ...

## Agreements
- ...

## Tensions
- ...

## Recommendation
- ...
```
