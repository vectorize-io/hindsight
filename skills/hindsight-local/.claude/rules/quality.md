---
description: Quality assurance standards for security (OWASP), performance,
  accessibility (WCAG), and code quality reviews
---


# Quality Assurance Standards

## Core Rules

1. Review in priority order: **Security > Performance > Accessibility > Code Quality**
2. Every finding must include file:line, description, and fix
3. Severity levels: CRITICAL (security breach/data loss), HIGH (blocks launch), MEDIUM (this sprint), LOW (backlog)
4. Run automated tools first: `npm audit`, `bandit`, `lighthouse`
5. No false positives — every finding must be reproducible
6. Provide remediation code, not just descriptions
