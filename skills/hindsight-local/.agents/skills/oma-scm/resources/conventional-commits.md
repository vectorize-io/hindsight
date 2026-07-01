# Conventional Commits Guide

## Overview

Conventional Commits applies consistent rules to commit messages to enable:
- Automated CHANGELOG generation
- Semantic Versioning automation
- Improved commit history readability across teams

## Commit Message Structure

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

## Types

### Primary Types

| Type | Description | SemVer | Example |
|------|-------------|--------|---------|
| `feat` | Add new feature | MINOR | `feat: add user authentication` |
| `fix` | Bug fix | PATCH | `fix: resolve login timeout issue` |

### Secondary Types

| Type | Description | SemVer | Example |
|------|-------------|--------|---------|
| `docs` | Documentation changes | - | `docs: update API documentation` |
| `style` | Code style changes (formatting, semicolons, etc.) | - | `style: fix indentation` |
| `refactor` | Code improvement without behavior change | - | `refactor: extract helper function` |
| `perf` | Performance improvements | PATCH | `perf: optimize database queries` |
| `test` | Add/modify tests | - | `test: add unit tests for auth` |
| `chore` | Build, config, packages | - | `chore: update dependencies` |

## Scope

Scope indicates the area of changed code:

```
feat(auth): add OAuth2 support
fix(api): handle null response
refactor(ui): simplify button component
```

### Common Scopes
- `auth` - Authentication/authorization
- `api` - API endpoints
- `ui` - User interface
- `db` - Database
- `config` - Configuration
- `deps` - Dependencies

## Description

- **Imperative mood**: "add", "fix", "update" (NOT "added", "fixed", "updates")
- **Lowercase first letter**
- **No trailing period**
- **72 characters or less**

### Good Examples
```
feat(auth): add JWT token refresh mechanism
fix(api): handle empty response from payment gateway
refactor(ui): extract common button styles
```

### Bad Examples
```
feat(auth): Added JWT token refresh mechanism.  # past tense, period
fix: fix bug  # insufficient description
Update the authentication system to support OAuth2 tokens and refresh mechanism  # too long
```

## Body

Body is optional but useful for complex changes:

```
feat(auth): add multi-factor authentication

Implement TOTP-based two-factor authentication:
- Add QR code generation for authenticator apps
- Store encrypted TOTP secrets in database
- Add backup codes for account recovery

Closes #123
```

## Breaking Changes

Breaking changes marked with `!` or in footer:

```
feat(api)!: change response format for user endpoint

BREAKING CHANGE: The user endpoint now returns a nested object
instead of a flat structure. Update client code accordingly.
```

## Footer

### Issue References
```
feat(auth): add password reset flow

Closes #456
Refs #123, #789
```

### Co-Authors
```
feat(ui): redesign dashboard

Co-Authored-By: Jane Doe <jane@example.com>
Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

## Branch Naming Convention

| Type | Branch Prefix | Example |
|------|---------------|---------|
| feat | `feature/` | `feature/user-auth` |
| fix | `fix/` | `fix/login-timeout` |
| refactor | `refactor/` | `refactor/api-cleanup` |
| docs | `docs/` | `docs/api-guide` |
| hotfix | `hotfix/` | `hotfix/security-patch` |

## Commit Workflow

1. **Stage specific files** (NOT `git add .`):
   ```bash
   git add src/auth/login.ts
   git add tests/auth/login.test.ts
   ```

2. **Write commit message**:
   ```bash
   git commit -m "$(cat <<'EOF'
   feat(auth): add login rate limiting

   - Limit failed attempts to 5 per minute
   - Add exponential backoff for repeated failures
   - Log suspicious activity

   Closes #234

   Co-Authored-By: First Fluke <our.first.fluke@gmail.com>
   EOF
   )"
   ```

3. **Verify**:
   ```bash
   git log -1 --format=full
   ```

## Resources

- [Conventional Commits Specification](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)
