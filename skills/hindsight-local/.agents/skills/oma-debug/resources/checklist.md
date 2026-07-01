# Debug Agent - Self-Verification Checklist

Run through every item before submitting your fix.

## Fix Quality
- [ ] Root cause identified (not just symptom patched)
- [ ] Fix is minimal and focused (no unrelated changes)
- [ ] Edge cases handled (null, empty, boundary values)
- [ ] No new bugs introduced

## Regression Test
- [ ] Test written that reproduces the bug
- [ ] Fix reverted → test fails (confirms test actually catches the bug)
- [ ] Fix restored → test passes
- [ ] Test covers the specific triggering condition
- [ ] All existing tests still pass

## Similar Patterns
- [ ] Searched for same pattern elsewhere in codebase
- [ ] Reported or fixed similar occurrences
- [ ] Noted if systemic issue needs architectural fix

## Documentation
- [ ] Bug report saved to `.agents/results/bugs/`
- [ ] Root cause clearly explained
- [ ] Prevention advice included
- [ ] Files modified listed

## Security Check (if applicable)
- [ ] Fix doesn't introduce SQL injection, XSS, or auth bypass
- [ ] Sensitive data not exposed in error messages or logs
