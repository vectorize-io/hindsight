# Methodology Selection Guide

Choose the lightest method that fits the problem.

## 1. Diagnostic Mode

Use when:
- the complaint is vague
- the user feels pain but cannot name the architecture issue
- it is unclear whether the problem is architecture, planning, QA, or debugging

Output:
- clarified problem statement
- why this is an architecture concern
- recommended next mode

## 2. Recommendation Mode

Use when:
- the decision is concrete
- the user wants an architectural direction
- the analysis does not require formal quality-attribute scenarios

Output:
- options
- tradeoff comparison
- recommendation

## 3. Design-Twice Mode

Use when:
- the decision is structural and likely to harden into the codebase
- the first obvious option may anchor the team too early
- interface, boundary, or decomposition choices are material

Output:
- two or more materially different options
- comparison and synthesis

## 4. ATAM-style Mode

Use when:
- the user asks for architecture review, tradeoffs, risks, or quality attributes
- reliability, performance, modifiability, security, or operability are central
- scenario-based evaluation is needed

Output:
- scenarios
- sensitivity points
- tradeoff points
- risks and non-risks

## 5. CBAM-style Mode

Use when:
- the user asks what to prioritize or invest in first
- the decision is economic or sequencing-oriented
- multiple architecture improvements compete for time and budget

Output:
- cost/benefit comparison
- sequencing recommendation

## 6. ADR Mode

Use when:
- the analysis is done and the team needs a portable record
- a concise decision artifact is the deliverable

Output:
- context
- decision
- alternatives
- consequences

## Selection Heuristics

- Start with Diagnostic Mode if the problem is fuzzy
- Default to Recommendation Mode if the decision is clear and moderate in scope
- Upgrade to Design-Twice Mode for material structural choices
- Upgrade to ATAM-style Mode for quality-attribute and risk-heavy reviews
- Upgrade to CBAM-style Mode for investment prioritization
- Finish with ADR Mode when a durable decision record is useful
