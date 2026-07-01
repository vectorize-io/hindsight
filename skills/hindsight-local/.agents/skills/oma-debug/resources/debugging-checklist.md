# Debugging Checklist

Use this checklist when investigating bugs to ensure thorough analysis.

## Initial Information Gathering

- [ ] **Bug description** - What is the expected vs actual behavior?
- [ ] **Error messages** - Exact error text, stack trace, error codes
- [ ] **Reproduction steps** - Clear, numbered steps to trigger the bug
- [ ] **Frequency** - Every time, intermittent, specific conditions?
- [ ] **Environment** - Browser/OS/version, mobile device, server environment
- [ ] **User impact** - How many users affected? Severity?
- [ ] **Recent changes** - New deploy? Code changes? Configuration updates?
- [ ] **Screenshots/videos** - Visual evidence of the bug

## Frontend Debugging

### JavaScript/TypeScript Errors

- [ ] Check browser console (F12) for errors
- [ ] Review stack trace - which file and line?
- [ ] Identify error type (TypeError, ReferenceError, etc.)
- [ ] Check for null/undefined values
- [ ] Verify variable types (especially after API calls)
- [ ] Look for typos in property names

### React Component Issues

- [ ] **State problems**
  - [ ] State not updating? (need setState/useState)
  - [ ] Stale state in closures? (use refs or functional updates)
  - [ ] State out of sync with props?

- [ ] **Effect problems**
  - [ ] Missing dependencies in useEffect?
  - [ ] Effect running too many times?
  - [ ] Cleanup function needed?
  - [ ] Memory leak from uncancelled requests?

- [ ] **Rendering issues**
  - [ ] Conditional rendering logic correct?
  - [ ] Keys on list items unique and stable?
  - [ ] Unnecessary re-renders? (use React DevTools Profiler)

### UI/UX Issues

- [ ] CSS not applying? (check specificity, typos)
- [ ] Layout broken? (check responsive breakpoints)
- [ ] Accessibility issues? (keyboard nav, screen reader)
- [ ] Dark mode broken?
- [ ] Mobile-specific issues?

### API Integration

- [ ] **Network tab investigation**
  - [ ] Request sent? (check Headers tab)
  - [ ] Correct endpoint URL?
  - [ ] Proper HTTP method (GET, POST, etc.)?
  - [ ] Headers included (Authorization, Content-Type)?
  - [ ] Request body formatted correctly?
  - [ ] Response status code (200, 401, 404, 500)?
  - [ ] Response data structure matches expected?

- [ ] **Error handling**
  - [ ] Try/catch around API calls?
  - [ ] Error state displayed to user?
  - [ ] Timeout handling?
  - [ ] Network failure handling?

- [ ] **Loading states**
  - [ ] Loading spinner shown?
  - [ ] Disabled buttons during load?
  - [ ] Optimistic updates causing issues?

## Backend Debugging

### Python/FastAPI Errors

- [ ] **Stack trace analysis**
  - [ ] Read error message carefully
  - [ ] Identify the failing file and line
  - [ ] Understand the error type (ValueError, KeyError, etc.)
  - [ ] Check for None values
  - [ ] Verify data types

- [ ] **Database issues**
  - [ ] Query syntax correct?
  - [ ] N+1 query problem?
  - [ ] Missing JOIN?
  - [ ] Indexes present on filtered columns?
  - [ ] Connection pool exhausted?
  - [ ] Transaction isolation issues?

- [ ] **Authentication/Authorization**
  - [ ] Token validation working?
  - [ ] Token expired?
  - [ ] Permissions checked correctly?
  - [ ] User session valid?
  - [ ] CORS configured properly?

### API Endpoint Issues

- [ ] **Request validation**
  - [ ] Pydantic schema matches request?
  - [ ] Required fields present?
  - [ ] Field types correct?
  - [ ] Value ranges validated?

- [ ] **Response formatting**
  - [ ] Correct status code?
  - [ ] Response schema matches contract?
  - [ ] Error responses formatted properly?
  - [ ] Pagination working?

- [ ] **Performance**
  - [ ] Response time acceptable (<200ms)?
  - [ ] Query optimization needed?
  - [ ] Caching implemented?
  - [ ] Rate limiting working?

### Server Issues

- [ ] **Logs examination**
  - [ ] Check application logs
  - [ ] Check error logs
  - [ ] Check access logs
  - [ ] Timestamps correlate with bug reports?

- [ ] **Environment**
  - [ ] Environment variables set?
  - [ ] Configuration correct for environment (dev/prod)?
  - [ ] External services reachable?
  - [ ] Database migrations applied?

## Mobile Debugging

### Platform-Specific Issues

- [ ] **iOS vs Android differences**
  - [ ] Test on both platforms
  - [ ] Check platform-specific code (Platform.OS)
  - [ ] Review native module integration
  - [ ] Verify permissions (camera, location, etc.)

- [ ] **Device-specific**
  - [ ] Test on actual device (not just emulator)
  - [ ] Check different screen sizes
  - [ ] Test different OS versions
  - [ ] Memory constraints on older devices?

### Flutter-Specific

- [ ] **Widget tree issues**
  - [ ] State management correct? (Riverpod/Bloc)
  - [ ] Build method pure?
  - [ ] Keys used properly?
  - [ ] Dispose called for controllers?

- [ ] **Platform channels**
  - [ ] Native code errors?
  - [ ] Method channel names correct?
  - [ ] Data serialization working?

### Mobile Performance

- [ ] Memory leaks? (listeners not disposed)
- [ ] Excessive rebuilds?
- [ ] Large images not optimized?
- [ ] Too many simultaneous network requests?

## Security Bugs

- [ ] **Authentication bypassed?**
  - [ ] Token validation on all protected routes?
  - [ ] Token expiry checked?
  - [ ] Refresh token flow secure?

- [ ] **Authorization broken?**
  - [ ] User can access others' data?
  - [ ] Admin functions accessible to regular users?
  - [ ] Horizontal privilege escalation possible?

- [ ] **Injection vulnerabilities**
  - [ ] SQL injection? (use parameterized queries)
  - [ ] XSS? (sanitize user input)
  - [ ] Command injection?
  - [ ] Path traversal?

- [ ] **Data exposure**
  - [ ] Sensitive data in logs?
  - [ ] Passwords in plain text?
  - [ ] API keys exposed?
  - [ ] Error messages leaking info?

## Performance Bugs

### Frontend Performance

- [ ] **Lighthouse audit**
  - [ ] Performance score >90?
  - [ ] First Contentful Paint <1.5s?
  - [ ] Largest Contentful Paint <2.5s?
  - [ ] Cumulative Layout Shift <0.1?

- [ ] **Bundle analysis**
  - [ ] Main bundle <500KB?
  - [ ] Code splitting implemented?
  - [ ] Large dependencies tree-shaken?
  - [ ] Lazy loading used?

- [ ] **React performance**
  - [ ] Unnecessary re-renders? (React DevTools Profiler)
  - [ ] Memo/useMemo/useCallback used appropriately?
  - [ ] Virtual scrolling for long lists?

### Backend Performance

- [ ] **API latency**
  - [ ] p95 response time <200ms?
  - [ ] Slow queries identified?
  - [ ] Indexes added?
  - [ ] Caching implemented?

- [ ] **Database**
  - [ ] Connection pooling configured?
  - [ ] Query plan optimized? (EXPLAIN)
  - [ ] N+1 queries eliminated?

## Root Cause Analysis

- [ ] **Reproduce the bug**
  - [ ] Follow exact reproduction steps
  - [ ] Confirm bug exists
  - [ ] Note any variations

- [ ] **Isolate the problem**
  - [ ] Which component/function fails?
  - [ ] What input triggers it?
  - [ ] What conditions must be met?

- [ ] **Trace the data flow**
  - [ ] Where does the bad data come from?
  - [ ] How is it transformed?
  - [ ] Where does it cause the failure?

- [ ] **Identify the root cause**
  - [ ] Not just the symptom
  - [ ] Why does this happen?
  - [ ] What assumption was wrong?

## Fix Verification

- [ ] **Fix applied**
  - [ ] Code changed in correct file(s)
  - [ ] Logic correct
  - [ ] Edge cases handled
  - [ ] No new bugs introduced

- [ ] **Testing**
  - [ ] Regression test written
  - [ ] Test passes
  - [ ] Manual verification done
  - [ ] Edge cases tested

- [ ] **Related areas checked**
  - [ ] Similar patterns elsewhere?
  - [ ] Other code depending on this?
  - [ ] Breaking changes avoided?

- [ ] **Documentation**
  - [ ] Bug documented in Knowledge Base
  - [ ] Fix explained
  - [ ] Prevention notes added

## When to Escalate

Escalate to other agents if:

- [ ] Bug spans multiple domains → **oma-coordination**
- [ ] Requires architectural change → **PM Agent**
- [ ] Need comprehensive security audit → **QA Agent**
- [ ] Complex frontend refactoring needed → **Frontend Agent**
- [ ] Database schema changes needed → **Backend Agent**
- [ ] Platform-specific mobile issue → **Mobile Agent**

## Priority Assessment

**CRITICAL** - Fix immediately:
- [ ] App crashes on launch
- [ ] Data loss or corruption
- [ ] Security vulnerability
- [ ] Payment/auth completely broken
- [ ] Affects all users

**HIGH** - Fix within 24 hours:
- [ ] Major feature broken
- [ ] Affects >50% of users
- [ ] No workaround available
- [ ] Significant revenue impact

**MEDIUM** - Fix within sprint:
- [ ] Minor feature broken
- [ ] Affects <50% of users
- [ ] Workaround exists
- [ ] Moderate inconvenience

**LOW** - Schedule for future:
- [ ] Edge case
- [ ] Cosmetic issue
- [ ] Rarely encountered
- [ ] No user impact

## Documentation Template

After fixing, document in `.agents/results/bugs/`:

```markdown
# Bug: [Title]
**Date**: YYYY-MM-DD
**Severity**: CRITICAL/HIGH/MEDIUM/LOW
**Status**: FIXED

## Problem
[What was broken]

## Root Cause
[Why it was broken]

## Fix
[What was changed]

## Files Modified
- path/to/file

## Testing
- [x] Regression test added
- [x] Manual verification
- [x] Related areas checked

## Prevention
[How to avoid in future]
```

---

## Pro Tips

1. **Read the error message** - It usually tells you exactly what's wrong
2. **Reproduce first** - Don't waste time fixing unconfirmed bugs
3. **One change at a time** - Don't fix multiple things simultaneously
4. **Test thoroughly** - Verify the fix and check for regressions
5. **Document everything** - Future you will be grateful
6. **Look for patterns** - One bug often reveals more

## Tools Reference

- **Browser DevTools**: F12 (Console, Network, React DevTools)
- **Serena MCP**: find_symbol, search_for_pattern, find_referencing_symbols
- **Antigravity Browser**: Automated testing and reproduction
- **React Profiler**: Performance analysis
- **Lighthouse**: Performance audit
- **Git bisect**: Find when bug was introduced
