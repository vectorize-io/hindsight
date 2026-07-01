---
description: Conventional Commits specification and git commit workflow rules
globs:
alwaysApply: false
---

# Commit Standards

## Format

```
<type>(<scope>): <description>

[optional body]
```

**Types**: feat, fix, refactor, docs, test, chore, style, perf

## Rules

1. **Analyze changes** before committing
2. **Split by feature** if changes span multiple domains
3. **Description** under 72 characters, imperative mood, lowercase, no trailing period
4. **Never** use `git add -A` or `git add .` without explicit permission
5. **Never** commit files that may contain secrets (.env, credentials)
6. **Always** use specific file names when staging
7. Split commits when: different scopes, different types, logically independent changes
8. Do NOT split when: all changes belong to a single feature, few files (5 or fewer)
