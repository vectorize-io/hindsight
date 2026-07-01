# Multi-Review Protocol

## Core Principle
**Every task is reviewed multiple times from different perspectives.**

---

## Review Types Guide

### 1. Completeness Review (Step 2)
- **Question**: "Is anything missing?"
- **Check**: Map requirements to plan items
- **Pass Condition**: All requirements reflected in plan

### 2. Meta Review (Step 3)
- **Question**: "Was the review done properly?"
- **Check**: Self-verify previous review was sufficient
- **Pass Condition**: No review gaps confirmed

### 3. Simplicity Review (Step 4)
- **Question**: "Is this over-engineered?"
- **Check**: Question necessity of each component
- **Remove**: "Might need later", speculative features

### 4. Alignment Review (Step 6)
- **Question**: "Did we build what was requested?"
- **Check**: Compare plan vs implementation
- **Pass Condition**: 1:1 mapping confirmed

### 5. Safety Review (Step 7)
- **Question**: "Is there anything dangerous?"
- **Check**: OWASP Top 10, potential bugs
- **Tools**: npm audit, bandit, lighthouse
- **Pass Condition**: Zero CRITICAL/HIGH issues

### 6. Regression Review (Step 8)
- **Question**: "Did improvements break anything?"
- **Check**: Existing tests pass, existing features work
- **Pass Condition**: No regressions

### 7. Reusability Review (Step 10)
- **Question**: "Can we leverage existing code?"
- **Check**: Similar functions/components exist
- **Action**: Integrate if reusable

### 8. Consistency Review (Step 12)
- **Question**: "Is everything harmonious?"
- **Check**: Naming, style, architecture consistency
- **Pass Condition**: Aligns with existing codebase

### 9. Quality Review (Step 14)
- **Question**: "Does it meet quality standards?"
- **Check**: lint, types, coverage, complexity
- **Pass Condition**: All quality metrics pass

### 10. Cascade Impact Review (Step 16)
- **Question**: "Did we break anything elsewhere?"
- **Check**: Use find_referencing_symbols for impact scope
- **Pass Condition**: No cascade impact or handled

### 11. Final Review (Step 17)
- **Question**: "Is this ready to deploy?"
- **Check**: Complete checklist final verification
- **Pass Condition**: User final approval

---

## Failure Recovery

| Review | Return Point on Failure |
|--------|------------------------|
| Step 2-4 | Step 1 (Revise plan) |
| Step 6-8 | Step 5 (Fix implementation) |
| Step 10-13 | Step 9 (Restart refinement) |
| Step 14-17 | Appropriate phase based on failure |
