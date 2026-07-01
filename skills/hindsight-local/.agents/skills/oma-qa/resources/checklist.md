# QA Review Checklist

## Security Checklist

### Authentication & Authorization
- [ ] Passwords hashed with bcrypt/argon2 (not MD5/SHA1)
- [ ] Password strength requirements enforced (min 8 chars)
- [ ] JWT tokens properly signed and validated
- [ ] Refresh tokens implemented (if long sessions needed)
- [ ] Token expiry reasonable (15min access, 7day refresh)
- [ ] Authorization checks on ALL endpoints
- [ ] Users can only access their own data
- [ ] Admin functions require admin role
- [ ] Rate limiting on auth endpoints (5-10 attempts/min)
- [ ] Account lockout after failed login attempts (optional)
- [ ] MFA available (optional, but recommended)

### Input Validation & Injection
- [ ] SQL injection: ORM used OR parameterized queries
- [ ] XSS: Input sanitized, CSP headers set
- [ ] Command injection: No shell execution with user input
- [ ] Path traversal: File paths validated
- [ ] LDAP injection: LDAP queries parameterized
- [ ] XML injection: XML parsing secure
- [ ] Email validation (proper regex/library)
- [ ] URL validation (allowlist for external requests)

### Data Protection
- [ ] HTTPS enforced (redirect HTTP to HTTPS)
- [ ] Sensitive data NOT in logs
- [ ] Sensitive data NOT in error messages
- [ ] Sensitive data NOT in URLs (use POST body)
- [ ] Database backups encrypted
- [ ] PII data encrypted at rest (if applicable)
- [ ] Secure session management (httpOnly, secure, sameSite cookies)

### API Security
- [ ] CORS properly configured (not `*` in production)
- [ ] CSRF protection enabled
- [ ] Rate limiting on API endpoints
- [ ] API keys/tokens NOT in source code
- [ ] API versioning implemented
- [ ] Proper error handling (no stack traces exposed)

### Dependencies
- [ ] No high/critical vulnerabilities (npm audit / safety check)
- [ ] Dependencies up-to-date
- [ ] No unused dependencies
- [ ] License compliance checked

---

## Performance Checklist

### Backend Performance
- [ ] API response time < 200ms (p95)
- [ ] Database queries optimized (no N+1)
- [ ] Database indexes on foreign keys and frequent queries
- [ ] Connection pooling configured
- [ ] Caching implemented (Redis for frequent queries)
- [ ] Pagination for large result sets
- [ ] Async operations where appropriate
- [ ] Background jobs for heavy tasks

### Frontend Performance
- [ ] Lighthouse Performance score > 90
- [ ] First Contentful Paint (FCP) < 1.5s
- [ ] Largest Contentful Paint (LCP) < 2.5s
- [ ] Cumulative Layout Shift (CLS) < 0.1
- [ ] Time to Interactive (TTI) < 3.5s
- [ ] Bundle size < 500KB (main bundle)
- [ ] Code splitting implemented
- [ ] Lazy loading for non-critical components
- [ ] Images optimized (WebP, compression)
- [ ] Images lazy loaded (loading="lazy")
- [ ] Fonts optimized (font-display: swap)
- [ ] No render-blocking resources
- [ ] Service worker for caching (optional)

### Mobile Performance
- [ ] App size < 30MB (Android), < 50MB (iOS)
- [ ] Cold start < 2s
- [ ] Smooth scrolling (60fps)
- [ ] No memory leaks
- [ ] Battery usage minimal
- [ ] Offline support (if required)

---

## Accessibility Checklist (WCAG 2.2 AA)

### Perceivable
- [ ] All images have alt text
- [ ] Decorative images have empty alt (`alt=""`)
- [ ] Color contrast 4.5:1 (normal text), 3:1 (large text)
- [ ] Text resizable up to 200% without loss of content
- [ ] Content understandable without color alone
- [ ] Audio/video has captions (if applicable)

### Operable
- [ ] All functionality available via keyboard
- [ ] No keyboard trap
- [ ] Focus order is logical
- [ ] Focus indicators visible
- [ ] Skip to main content link
- [ ] No content flashes more than 3 times per second
- [ ] Enough time to read/interact with content
- [ ] Pause/stop for moving content

### Understandable
- [ ] Page language set (`<html lang="en">`)
- [ ] Clear labels on form inputs
- [ ] Error messages clear and helpful
- [ ] Required fields indicated
- [ ] Consistent navigation across pages
- [ ] Predictable behavior (no unexpected popups)

### Robust
- [ ] Valid HTML (semantic tags)
- [ ] ARIA labels where needed
- [ ] ARIA roles appropriate
- [ ] Works with screen readers (test with NVDA/JAWS)
- [ ] Works in different browsers (Chrome, Firefox, Safari, Edge)

---

## Testing Checklist

### Unit Tests
- [ ] Test coverage > 80%
- [ ] All business logic functions tested
- [ ] Edge cases covered
- [ ] Error handling tested
- [ ] Mocks used appropriately
- [ ] Tests run fast (< 10s total)
- [ ] No flaky tests

### Integration Tests
- [ ] All API endpoints tested
- [ ] Database operations tested
- [ ] Auth flow tested
- [ ] Error responses tested (401, 403, 404, 500)
- [ ] Request validation tested

### E2E Tests
- [ ] Critical user flows tested (registration, login, main feature)
- [ ] Happy path tested
- [ ] Error scenarios tested
- [ ] Mobile responsive tested
- [ ] Cross-browser tested (Chrome, Firefox, Safari)

### Test Governance
- [ ] Test levels and scope are defined clearly
- [ ] Important requirements trace to test cases or test scenarios
- [ ] Entry / exit criteria are defined for major release decisions
- [ ] Test design technique is appropriate for risk and feature type

### Performance Tests
- [ ] Load testing (1000 concurrent users)
- [ ] Stress testing (identify breaking point)
- [ ] Database under load tested
- [ ] API rate limits tested

---

## Code Quality Checklist

### Architecture
- [ ] Clear separation of concerns
- [ ] DRY principle followed (no duplication > 5%)
- [ ] SOLID principles followed
- [ ] Dependency injection used
- [ ] Repository pattern (backend)
- [ ] Component composition (frontend)

### Code Metrics
- [ ] Cyclomatic complexity < 10 per function
- [ ] Function length < 50 lines
- [ ] File length < 500 lines
- [ ] No deeply nested code (< 4 levels)
- [ ] Meaningful variable names

### Error Handling
- [ ] All async operations have try/catch
- [ ] Errors logged appropriately
- [ ] User-friendly error messages
- [ ] No silent failures
- [ ] Graceful degradation

### Documentation
- [ ] README with setup instructions
- [ ] API documentation (OpenAPI/Swagger)
- [ ] Complex logic documented
- [ ] Environment variables documented
- [ ] No TODO/FIXME in production code

---

## ISO Quality Alignment

### ISO/IEC 25010
- [ ] Functional suitability considered
- [ ] Performance efficiency considered
- [ ] Compatibility considered where integration matters
- [ ] Usability / accessibility considered
- [ ] Reliability considered
- [ ] Security considered
- [ ] Maintainability considered
- [ ] Portability considered when relevant

### ISO/IEC 29119
- [ ] Test strategy or test plan exists for significant changes
- [ ] Test basis, test conditions, and expected results are clear
- [ ] Risk-based prioritization is visible in test scope
- [ ] Test evidence and traceability are sufficient for review or audit

---

## Browser Compatibility Checklist

### Desktop
- [ ] Chrome (latest 2 versions)
- [ ] Firefox (latest 2 versions)
- [ ] Safari (latest 2 versions)
- [ ] Edge (latest 2 versions)

### Mobile
- [ ] iOS Safari (latest 2 versions)
- [ ] Android Chrome (latest 2 versions)
- [ ] Responsive breakpoints (320px, 768px, 1024px, 1440px)

---

## DevOps Checklist

### Environment
- [ ] Environment variables used (not hardcoded)
- [ ] .env.example provided
- [ ] Secrets NOT in source code
- [ ] Different configs for dev/staging/prod

### Logging
- [ ] Appropriate log levels (DEBUG, INFO, WARNING, ERROR)
- [ ] No sensitive data in logs
- [ ] Structured logging (JSON format)
- [ ] Log rotation configured

### Monitoring
- [ ] Health check endpoint (`/health`)
- [ ] Error tracking (Sentry, Rollbar, etc.)
- [ ] Performance monitoring (APM)
- [ ] Uptime monitoring

### Deployment
- [ ] CI/CD pipeline configured
- [ ] Automated tests in CI
- [ ] Database migrations automated
- [ ] Rollback plan documented
- [ ] Zero-downtime deployment (if required)

---

## Final Sign-Off

### Critical (Must Pass)
- [ ] No CRITICAL security vulnerabilities
- [ ] No HIGH security vulnerabilities
- [ ] All E2E tests passing
- [ ] Performance meets requirements
- [ ] No data loss scenarios

### Important (Should Pass)
- [ ] Test coverage > 80%
- [ ] Accessibility WCAG 2.2 AA
- [ ] Code quality metrics met
- [ ] Documentation complete

### Nice-to-Have (Can Address Later)
- [ ] Code refactoring opportunities documented
- [ ] Performance optimization ideas documented
- [ ] Future enhancement ideas documented

---

## Issue Prioritization

### CRITICAL (Block Deployment)
- Security vulnerabilities (SQL injection, XSS, auth bypass)
- Data loss bugs
- Application crashes
- Complete feature breakage

### HIGH (Fix Before Launch)
- Performance issues (> 5s load time)
- Major accessibility violations
- Missing auth checks
- Broken core functionality

### MEDIUM (Fix in Sprint)
- Minor bugs
- Code quality issues
- Missing tests
- Minor accessibility issues

### LOW (Backlog)
- Refactoring opportunities
- Performance optimizations
- Nice-to-have features
- Documentation improvements

---

## Notes

- Run automated tools FIRST: `npm audit`, `bandit`, `lighthouse`
- Use Serena MCP for code analysis patterns
- Use Chrome DevTools MCP (`new_page` with `isolatedContext: "qa-test"`) for runtime verification and E2E testing
- Document all findings with file:line references
- Provide remediation code examples
- Estimate fix time for each issue

---

## Runtime Verification (after static review)

Record results in the structured table format defined in `execution-protocol.md` Step 2.5 (Recording Results).

- [ ] Application starts without errors
- [ ] All modified endpoints return expected status codes; verify with `list_network_requests()`
- [ ] Form submissions produce correct database state; verify with `fill_form()` + `list_network_requests()`
- [ ] Error states render user-friendly messages (not stack traces); verify with `take_snapshot()` on error paths
- [ ] Empty/loading/error UI states all handled; verify with `navigate_page()` to empty state routes + `take_snapshot()`
- [ ] Interactive elements respond to input (not display-only); verify with `click(uid)` + `take_snapshot()` before/after
- [ ] Auth flows work end-to-end (register → login → protected route → logout); verify with sequential `fill()` + `click()` + `list_network_requests()`
- [ ] Rate limiting / throttling triggers at configured thresholds; verify with rapid `evaluate_script(fetch)` calls
- [ ] File upload/download actually transfers data (not stubbed); verify with `upload_file(uid, filePath)` + response check
- [ ] Pagination returns correct pages (not always page 1); verify with `click()` page 2 + `take_snapshot()` to confirm different content
- [ ] Zero JS console errors on critical paths; verify with `list_console_messages(types: ["error"])`
