# CODEOWNERS Playbook (SCM)

Use this checklist to keep ownership and approval paths healthy.

## What to check

1. `CODEOWNERS` exists in one supported location:
   - `.github/CODEOWNERS`
   - `CODEOWNERS` (repo root)
   - `docs/CODEOWNERS`
2. Critical paths are explicitly owned (avoid relying only on `*` wildcard).
3. Owners in rules are active maintainers (not departed/inactive accounts).
4. Branch protection requires CODEOWNERS review for protected branches.
5. Rules are minimally overlapping and ordered intentionally.

## Baseline template

```text
# Global fallback
* @org/maintainers

# CI and release control
.github/workflows/* @org/devops
cli/commands/migrations/* @org/release-engineering

# Agent system and governance
.agents/hooks/** @org/agent-platform
.agents/workflows/** @org/agent-platform

# Product docs/content
web/content/** @org/docs
docs/** @org/docs
```

## Common anti-patterns

- Only global `*` rule with no critical-path overrides.
- Owners set to individuals only (no team alias).
- Stale owners after team changes.
- Critical infra/security paths missing explicit owners.

## Review cadence

- Re-check CODEOWNERS at least once per quarter.
- Re-check immediately after org/team restructure.
