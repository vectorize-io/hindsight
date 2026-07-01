# QA Agent - ISO Quality Guide

Use this file when the user asks for standards-based QA suggestions, enterprise audit readiness, or process-oriented quality improvements.

## Positioning

- Use **ISO/IEC 25010** as the practical software quality model.
- Treat **ISO/IEC 25010** as part of the **ISO/IEC 25000 SQuaRE family** when users say "ISO 25000".
- Use **ISO/IEC 29119** for test process, test design, traceability, and evidence recommendations.
- Do not force standards language into every QA review. Add it when it improves decision quality, auditability, or delivery discipline.

## 1. ISO/IEC 25010 Mapping

Map findings to these quality characteristics when relevant:

- functional suitability
- performance efficiency
- compatibility
- usability
- reliability
- security
- maintainability
- portability

Use this mapping to answer:

- Which quality areas are under-covered?
- Which areas are blocking release confidence?
- Which missing tests leave important quality characteristics unverified?

## 2. ISO/IEC 29119 Suggestions

Recommend these when test discipline is weak:

- test strategy or test plan
- test levels and scope definition
- requirement / risk to test traceability
- explicit test conditions and expected results
- risk-based prioritization
- entry / exit criteria
- evidence retention for review or audit

## 3. When to Mention Standards

Mention ISO alignment when:

- the user asks for audit readiness
- the work is enterprise or regulated
- release governance is unclear
- testing exists but lacks structure or traceability
- quality discussion is too vague and needs a common model

Avoid overusing standards language when:

- the task is a quick bug fix
- the user wants only concrete defects
- standards language would add ceremony without action

## 4. Review Output Pattern

If standards are relevant, add a short section:

```md
## Standards Alignment

### ISO/IEC 25010
- Quality characteristic:
- Gap:
- Recommendation:

### ISO/IEC 29119
- Test governance gap:
- Traceability / evidence gap:
- Recommendation:
```

## 5. Guardrails

- Prefer practical suggestions over compliance theater
- Do not claim formal compliance unless evidence exists
- Tie every standards suggestion to a delivery, test, or risk outcome
- Keep standards comments secondary to concrete findings
