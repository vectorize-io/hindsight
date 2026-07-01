# QA Agent - Self-Check

Verify your own review quality before submitting the QA report.

## Completeness
- [ ] All files in scope were reviewed (none skipped)
- [ ] Security section covers OWASP Top 10 categories
- [ ] Performance metrics include actual numbers vs. targets
- [ ] Accessibility check covers Perceivable, Operable, Understandable, Robust
- [ ] ISO/IEC 25010 and ISO/IEC 29119 suggestions were included when relevant

## Accuracy
- [ ] Every finding includes file path and line number
- [ ] Every finding is reproducible (not speculative)
- [ ] No false positives (double-check edge cases)
- [ ] Severity ratings are consistent (CRITICAL = data loss/security breach)

## Actionability
- [ ] Every finding has a specific remediation step
- [ ] Remediation code examples are correct and complete
- [ ] Priorities are clearly ordered (what to fix first)
- [ ] Estimated impact is noted for performance issues
- [ ] Standards-based suggestions are practical, not ceremonial

## Report Quality
- [ ] Overall status (PASS/WARNING/FAIL) matches findings
- [ ] Report is scannable (headers, bullets, status tags)
- [ ] No duplicate findings
- [ ] Sign-off checklist included for launch decisions
