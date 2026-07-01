# Architecture Agent - Examples

## Example 1: Recommendation Mode

**Request**: "Should this notifications subsystem become a separate service?"

**Good Output Shape**:
- Problem framing
- Current constraints
- Option A: keep in monolith, extract internal module
- Option B: dedicated service
- Comparison table
- Recommendation
- Risks and validation steps

## Example 2: ATAM-style Mode

**Request**: "Review this architecture for reliability and scaling tradeoffs."

**Good Output Shape**:
- Quality attribute scenarios
- Architectural approaches under review
- Sensitivity points
- Tradeoff points
- Risks / non-risks
- Prioritized recommendations

## Example 3: CBAM-style Mode

**Request**: "Which architecture refactor should we invest in first next quarter?"

**Good Output Shape**:
- Candidate investments
- Estimated benefit
- Estimated implementation cost
- Operational impact
- Priority ranking
- Recommended sequence

## Example 4: Diagnostic Mode

**Request**: "This code works, but every small change touches five files and I hate it."

**Good Output Shape**:
- Symptom summary
- Likely architecture cause
- Why this is an architecture problem instead of a bug/QA issue
- Selected next mode: Recommendation or ATAM-style

## Example 5: ADR Mode

**Request**: "Write an ADR for choosing event-driven processing over synchronous orchestration."

**Good Output Shape**:
- Context
- Decision
- Alternatives considered
- Consequences
- Follow-up validation
