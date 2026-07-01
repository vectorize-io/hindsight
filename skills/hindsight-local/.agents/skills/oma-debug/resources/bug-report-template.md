# Bug Report Template

Use this template when documenting bugs in the Knowledge Base.

Save to: `.agents/results/bugs/bug-YYYYMMDD-[short-description].md`

---

# Bug: [Short Descriptive Title]

**Date Reported**: YYYY-MM-DD
**Date Fixed**: YYYY-MM-DD (or "In Progress")
**Reporter**: [User name or issue number]
**Assignee**: [Agent that fixed it]
**Severity**: CRITICAL | HIGH | MEDIUM | LOW
**Status**: OPEN | IN PROGRESS | FIXED | ON HOLD | WON'T FIX

---

## Problem Description

**What happened?**
[Clear description of the bug from user's perspective]

**What was expected?**
[What should have happened instead]

**Impact**:
- Users affected: [All | Specific role | Percentage]
- Business impact: [Revenue loss | User frustration | Security risk]
- Workaround available: [Yes/No - describe if yes]

---

## Reproduction Steps

1. Navigate to [page/route]
2. Click on [button/element]
3. Enter [data] in [field]
4. Observe [unexpected behavior]

**Frequency**:
- [ ] Every time (100%)
- [ ] Intermittent (specify pattern: ___%)
- [ ] Specific conditions only (describe: ___)

---

## Evidence

**Error Messages**:
```
[Exact error text from console/logs]
```

**Stack Trace**:
```
[Full stack trace if available]
```

**Screenshots**:
- Before: [description or file path]
- After: [description or file path]
- Error state: [description or file path]

**Browser Console**:
```javascript
// Relevant console errors
```

**Network Requests**:
```
Request URL: [URL]
Status Code: [200/404/500/etc]
Response: [relevant response data]
```

---

## Environment

**Frontend**:
- Browser: [Chrome 120 | Firefox 121 | Safari 17]
- OS: [Windows 11 | macOS 14 | iOS 17 | Android 14]
- Screen size: [Desktop | Tablet | Mobile]
- Browser extensions: [Any relevant extensions]

**Backend**:
- Environment: [Development | Staging | Production]
- Server version: [API version]
- Database: [PostgreSQL 16.1]

**Mobile** (if applicable):
- Device: [iPhone 15 | Samsung Galaxy S24]
- OS Version: [iOS 17.2 | Android 14]
- App Version: [1.2.3]

---

## Investigation

### Initial Analysis

**Hypothesis**:
[What you think is causing the bug]

**Investigation Steps Taken**:
1. [What was checked]
2. [What was ruled out]
3. [Key findings]

### Root Cause

**Technical Explanation**:
[Deep dive into why the bug occurs]

**Code Location**:
- File: `path/to/file.tsx`
- Line: 145
- Function: `handleSubmit()`

**Specific Issue**:
```typescript
// Problem code
const user = data.user.profile.name; // Crashes if profile is undefined
```

**Why it happens**:
[Explain the conditions that trigger this]

---

## Solution

### Fix Applied

**Approach**:
[High-level description of the fix strategy]

**Code Changes**:

```typescript
// File: path/to/file.tsx (line 145)

// BEFORE (buggy code)
const user = data.user.profile.name;

// AFTER (fixed code)
const user = data?.user?.profile?.name ?? 'Unknown';
```

**Why this works**:
[Explain the fix]

### Files Modified

- `src/components/UserProfile.tsx` - Added null check for profile
- `src/lib/api/users.ts` - Improved error handling
- `src/components/UserProfile.test.tsx` - Added regression test

### Migration/Deployment Notes

**Database Changes**: None | [Describe migrations needed]
**Configuration Changes**: None | [Describe config updates]
**Breaking Changes**: None | [Describe breaking changes]
**Rollback Plan**: [How to revert if needed]

---

## Verification

### Testing Performed

- [x] **Regression test added**
  - File: `src/components/UserProfile.test.tsx`
  - Coverage: Null profile, undefined user, missing name

- [x] **Manual testing**
  - Tested original reproduction steps
  - Tested edge cases (null, undefined, empty)
  - Verified fix works in all browsers

- [x] **Related areas checked**
  - Found similar pattern in `TeamProfile.tsx` - also fixed
  - Checked all `.profile.` usages - 3 more locations updated

- [x] **Performance impact**: None | [Describe if any]

### Test Results

**Unit Tests**: 15/15 passing
**Integration Tests**: 8/8 passing
**E2E Tests**: 3/3 passing
**Manual QA**: Verified on Chrome, Firefox, Safari

---

## Prevention

### How to Avoid Similar Bugs

1. **Always check for null/undefined** before accessing nested properties
2. **Use optional chaining (`?.`)** for safe property access
3. **Provide default values** with nullish coalescing (`??`)
4. **Add TypeScript strict null checks** to catch at compile time
5. **Write defensive code** - assume data might be missing

### Code Patterns to Follow

```typescript
// GOOD: Safe access with fallback
const name = user?.profile?.name ?? 'Anonymous';

// GOOD: Explicit null check
if (user?.profile) {
  const name = user.profile.name;
}

// GOOD: Early return
if (!user?.profile) {
  return <div>No profile available</div>;
}

// BAD: Unsafe nested access
const name = user.profile.name; // Crashes if profile undefined
```

### Recommended Tools

- **TypeScript** with `strictNullChecks: true`
- **ESLint** rule: `no-unsafe-member-access`
- **Unit tests** for edge cases (null, undefined, empty)

---

## Related

**Similar Bugs**:
- Bug #123: Similar null check issue in `CommentList`
- Bug #456: Related data loading pattern

**Dependent Issues**:
- Issue #789: Need to improve error handling across all API calls

**Documentation**:
- [Link to architecture decision record]
- [Link to API documentation]

**External References**:
- [Stack Overflow link if used]
- [GitHub issue in library if relevant]

---

## Metrics

**Time to Fix**: [2 hours | 1 day | 1 week]
**Lines Changed**: [+15 -5]
**Files Affected**: [3 files]
**Tests Added**: [5 new tests]

---

## Communication

**Notified**:
- [x] Product Manager - Impact assessment
- [x] QA Team - Additional testing needed
- [x] Users affected - Via email/announcement
- [ ] Other teams - [Specify]

**Changelog Entry**:
```markdown
### Fixed
- Fixed crash when user profile is missing ([#issue-number])
```

---

## Lessons Learned

**What went well**:
- Quick identification of root cause
- Proactive fix of similar patterns
- Comprehensive test coverage

**What could improve**:
- Should have caught this in code review
- TypeScript strict mode would have prevented this
- Need better null check linting rules

**Action Items**:
- [ ] Enable TypeScript strictNullChecks across project
- [ ] Add ESLint rule for unsafe member access
- [ ] Update code review checklist to include null checks
- [ ] Create coding standards doc for defensive programming

---

## Tags

`frontend` `null-check` `crash` `typescript` `user-profile` `high-priority`

---

**Sign-off**: [Your name/agent name]
**Reviewed by**: [PM Agent | QA Agent | Frontend Agent]
**Approved for deploy**: [Yes/No]

---

## Template Notes

**How to use this template**:

1. **Copy this template** when documenting a new bug
2. **Fill in all sections** - don't skip parts
3. **Be specific** - future developers need details
4. **Include code** - show the before/after
5. **Link related issues** - help connect patterns
6. **Update status** as work progresses
7. **Save to Knowledge Base** for future reference

**Sections you can skip**:
- Screenshots (if text description is clear)
- Stack trace (if no error thrown)
- Migration notes (if no DB/config changes)

**Optional sections to add**:
- Timeline (for long-running bugs)
- Cost impact (for business-critical bugs)
- Vendor communication (if third-party bug)
