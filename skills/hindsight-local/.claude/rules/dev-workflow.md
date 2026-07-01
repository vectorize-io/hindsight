---
description: Monorepo developer workflow standards - mise tasks, git hooks,
  CI/CD, release automation
---


# Developer Workflow Standards

## Core Rules

1. Use `mise run` tasks instead of direct package manager commands
2. Run `mise install` after pulling changes that update runtime versions
3. Use parallel tasks for independent operations
4. Run lint/test only on apps with changed files
5. Validate commit messages with commitlint before committing
6. Configure CI to skip unchanged apps for faster builds
7. Check `mise tasks --all` to discover available tasks before running
8. Pin runtime versions in mise.toml for consistency
9. Never use direct package manager commands when mise tasks exist
10. Never modify mise.toml without understanding task dependencies
11. Never hardcode secrets in mise.toml files
12. Never skip reading task definitions before running unfamiliar tasks
