# QA Agent - Examples

## Example 1: Security Audit

**Input**: "Review the authentication system for security issues"

**Output**:
```
# QA Report: Authentication System
Status: FAIL

## Security Audit

### CRITICAL
1. **SQL Injection** - backend/api/users.py:45
   - Raw string interpolation in query
   - Fix: Use parameterized query or ORM

2. **Missing Rate Limiting** - backend/api/auth.py:23
   - Login endpoint has no rate limiting
   - Fix: Add @limiter.limit("5/minute")

### Passed
- [x] Passwords hashed with bcrypt (cost 12)
- [x] JWT properly signed with RS256
- [x] Refresh token rotation implemented
- [x] CORS configured (not wildcard)

## Recommendations
### High Priority (Before Launch)
1. Fix SQL injection in users.py:45
2. Add rate limiting to auth endpoints
3. Add account lockout after 10 failed attempts
```

## Example 2: Performance Review

**Input**: "Check frontend performance - users report slow load times"

**Output**:
```
# QA Report: Frontend Performance
Status: WARNING

## Performance
- Lighthouse Score: 67/100 [FAIL - Target > 90]
- FCP: 2.8s [FAIL - Target < 1.5s]
- LCP: 4.1s [FAIL - Target < 2.5s]
- Bundle: 1.2MB [FAIL - Target < 500KB]
- CLS: 0.05 [PASS]

## Root Causes
1. Full lodash import (72KB) - src/utils/helpers.ts:1
2. No code splitting on routes - src/app/layout.tsx
3. Unoptimized images (3 x 2MB PNGs) - public/images/

## Recommendations
### High Priority
1. Tree-shake lodash: `import uniq from 'lodash/uniq'`
2. Add dynamic imports: `const Page = dynamic(() => import(...))`
3. Convert images to WebP, add width/height, loading="lazy"

### Expected After Fix
- Bundle: ~350KB (-70%)
- FCP: ~1.2s
- Lighthouse: ~92
```

## Example 3: Full Pre-Launch Audit

**Input**: "Complete QA review before production deployment"

**Output**:
```
# QA Report: Pre-Launch Audit
Status: WARNING

## Summary
- Security: 0 CRITICAL, 1 HIGH, 2 MEDIUM
- Performance: PASS (all targets met)
- Accessibility: 1 HIGH, 3 MEDIUM
- Test Coverage: 83% [PASS]

## High Priority (Block Launch)
1. [SECURITY-HIGH] Missing CSRF protection on POST endpoints
2. [A11Y-HIGH] Login form has no aria-labels

## Medium Priority (Fix This Sprint)
1. [SECURITY-MED] API error responses expose stack traces
2. [SECURITY-MED] No Content-Security-Policy header
3. [A11Y-MED] Color contrast 3.8:1 on secondary text (need 4.5:1)
4. [A11Y-MED] Missing skip-to-content link
5. [A11Y-MED] Focus indicator invisible on dark theme

## Sign-Off
- [x] No CRITICAL vulnerabilities
- [ ] No HIGH vulnerabilities (2 remaining)
- [x] Performance targets met
- [x] Test coverage > 80%
```

## Example 4: Standards-Aligned QA Review

**Input**: "Review this release candidate and suggest ISO 25010 / 29119 improvements"

**Output**:
```
# QA Report: Release Candidate
Status: WARNING

## Standards Alignment

### ISO/IEC 25010
- Reliability gap: no chaos/failure-path validation for retry flows
- Maintainability gap: business rules duplicated in 3 services
- Security gap: privileged actions are not audit-logged consistently

### ISO/IEC 29119
- Missing requirement-to-test traceability for payment cancellation
- No explicit exit criteria for regression approval
- Edge-case test design is weak for invalid state transitions

## Recommendations
1. Add requirement-to-test mapping for payment, refund, and auth flows
2. Define release exit criteria: zero HIGH findings, all critical E2E paths green
3. Add state-transition and boundary-value tests for payment lifecycle
```
