---
name: oma-dev-workflow
description: Use when setting up or optimizing developer workflows in a monorepo, managing mise tasks, git hooks, CI/CD pipelines, database migrations, or release automation. Invoke for development environment setup, build automation, testing workflows, and release coordination.
---

# Dev Workflow - Monorepo Task Automation Specialist

## Scheduling

### Goal
Set up, run, optimize, and troubleshoot reproducible development workflows in monorepos using `mise`, task automation, validation pipelines, CI/CD, migrations, i18n builds, and release coordination.

### Intent signature
- User asks about dev servers, mise tasks, lint/format/typecheck/test/build, git hooks, CI/CD, migrations, generated clients, i18n builds, or release automation.
- User needs workflow execution or developer environment setup rather than product feature implementation.

### When to use

- Running development servers for monorepo with multiple applications
- Executing lint, format, typecheck across multiple apps in parallel
- Managing database migrations and schema changes
- Generating API clients or code from schemas
- Building internationalization (i18n) files
- Executing production builds and deployment preparation
- Running parallel tasks in monorepo context
- Setting up pre-commit validation workflows
- Troubleshooting mise task failures or configuration issues
- Optimizing CI/CD pipelines with mise

### When NOT to use

- Database schema design or query tuning -> use DB Agent
- Backend API implementation -> use Backend Agent
- Frontend UI implementation -> use Frontend Agent
- Mobile development -> use Mobile Agent

### Expected inputs
- Requested workflow operation, affected apps/packages, and current monorepo structure
- `mise.toml`, task definitions, CI files, migration/i18n/build configs, and failure logs when relevant
- Desired validation, setup, or release outcome

### Expected outputs
- Executed or documented mise task workflow
- Updated workflow config, CI/CD pipeline, hooks, env template, or release guidance when requested
- Status report with commands, outputs, failures, and next actions

### Dependencies
- `mise`, project task definitions, runtime versions, package managers behind mise tasks
- Resource guides for validation, database patterns, API workflows, i18n, release coordination, and troubleshooting

### Control-flow features
- Branches by affected apps, task dependency graph, port availability, task failure, and CI/release context
- Calls local process commands; may write workflow/config files
- Must avoid destructive tasks and secrets in workflow configs

## Structural Flow

### Entry
1. Identify affected apps/packages and requested workflow outcome.
2. Read `mise.toml` files and available tasks.
3. Determine whether tasks can run in parallel or must be sequential.

### Scenes
1. **PREPARE**: Analyze requirements, task graph, runtime prerequisites, and ports.
2. **ACQUIRE**: Inspect mise config, task definitions, CI hooks, env patterns, and logs.
3. **ACT**: Run or modify mise tasks, workflow configs, or validation pipelines.
4. **VERIFY**: Check exit codes, generated artifacts, logs, and CI compatibility.
5. **FINALIZE**: Report command status, duration, failures, and next steps.

### Transitions
- If runtime versions changed, run `mise install`.
- If changed-file tasks exist, prefer changed-scope validation.
- If port is occupied, resolve or select another port before starting dev server.
- If a task is unfamiliar, read its definition before running.

### Failure and recovery
- If task is missing, run `mise tasks --all`.
- If runtime is missing, run or recommend `mise install`.
- If task hangs, check for prompts or long-running dev-server behavior.
- If destructive task is requested, require confirmation.

### Exit
- Success: workflow runs or config changes are verified.
- Partial success: task failures, missing runtime, or CI/environment blockers are explicit.

## Logical Operations

### Actions
| Action | SSL primitive | Evidence |
|--------|---------------|----------|
| Read task definitions | `READ` | `mise.toml`, CI config |
| Select task strategy | `SELECT` | Parallel/sequential/changed-scope |
| Run workflow commands | `CALL_TOOL` | `mise run`, `mise install`, `mise tasks` |
| Write workflow config | `WRITE` | Hooks, CI, env templates |
| Validate outputs | `VALIDATE` | Exit codes, logs, artifacts |
| Report status | `NOTIFY` | Final workflow summary |

### Tools and instruments
- `mise`, shell, CI/CD tooling, project package manager tasks behind mise
- Resource guides for validation, database, API, i18n, release, and troubleshooting

### Canonical command path
```bash
mise tasks --all
mise install
mise run lint
mise run test
```

For app-specific tasks:
```bash
mise run //{path}:{task}
```

### Resource scope
| Scope | Resource target |
|-------|-----------------|
| `CODEBASE` | `mise.toml`, CI configs, scripts, generated clients |
| `LOCAL_FS` | Env templates, build outputs, logs |
| `PROCESS` | mise, build, test, lint, dev-server commands |
| `CREDENTIALS` | Secrets must not be hardcoded in workflow configs |

### Preconditions
- Target workflow and affected project area are identifiable.
- Task definitions can be discovered or missing-task state is reported.

### Effects and side effects
- May start dev servers, run tests/builds, generate clients, run migrations, or edit workflow configs.
- May consume CPU/time or occupy ports.

### Guardrails

1. Always use `mise run` tasks instead of direct package manager commands
2. Run `mise install` after pulling changes that might update runtime versions
3. Use parallel tasks (`mise run lint`, `mise run test`) for independent operations
4. Run lint/test only on apps with changed files (`lint:changed`, `test:changed`)
5. Validate commit messages with commitlint before committing
6. Run pre-commit validation pipeline for staged files only
7. Configure CI to skip unchanged apps for faster builds
8. Check `mise tasks --all` to discover available tasks before running
9. Verify task output and exit codes for CI/CD integration
10. Document task dependencies in mise.toml comments
11. Use consistent task naming conventions across apps
12. Enable mise in CI/CD pipelines for reproducible builds
13. Pin runtime versions in mise.toml for consistency
14. Test tasks locally before committing CI/CD changes
15. Never use direct package manager commands when mise tasks exist
16. Never modify mise.toml without understanding task dependencies
17. Never skip `mise install` after toolchain version updates
18. Never run dev servers without checking port availability first
19. Never commit without running validation on affected apps
20. Never ignore task failures - always investigate root cause
21. Never hardcode secrets in mise.toml files
22. Never assume task availability - always verify with `mise tasks`
23. Never run destructive tasks (clean, reset) without confirmation
24. Never skip reading task definitions before running unfamiliar tasks

### Technical Guidelines

### Prerequisites

```bash
# Install mise
curl https://mise.run | sh

# Activate in shell
echo 'eval "$(~/.local/bin/mise activate)"' >> ~/.zshrc

# Install all runtimes defined in mise.toml
mise install

# Verify installation
mise list
```

### Project Structure (Monorepo)

```
project-root/
├── mise.toml            # Root task definitions
├── apps/
│   ├── api/            # Backend application
│   │   └── mise.toml   # App-specific tasks
│   ├── web/            # Frontend application
│   │   └── mise.toml
│   └── mobile/         # Mobile application
│       └── mise.toml
├── packages/
│   ├── shared/         # Shared libraries
│   └── config/         # Shared configuration
└── scripts/            # Utility scripts
```

### Task Syntax

**Root-level tasks:**
```bash
mise run lint        # Lint all apps (parallel)
mise run test        # Test all apps (parallel)
mise run dev         # Start all dev servers
mise run build       # Production builds
```

**App-specific tasks:**
```bash
# Syntax: mise run //{path}:{task}
mise run //apps/api:dev
mise run //apps/api:test
mise run //apps/web:build
```

### Common Task Patterns

| Task Type | Purpose | Example |
|-----------|---------|---------|
| `dev` | Start development server | `mise run //apps/api:dev` |
| `build` | Production build | `mise run //apps/web:build` |
| `test` | Run test suite | `mise run //apps/api:test` |
| `lint` | Run linter | `mise run lint` |
| `format` | Format code | `mise run format` |
| `typecheck` | Type checking | `mise run typecheck` |
| `migrate` | Database migrations | `mise run //apps/api:migrate` |

### Reference Guide

| Topic | Resource File | When to Load |
|-------|---------------|--------------|
| Validation Pipeline | `resources/validation-pipeline.md` | Git hooks, CI/CD, change-based testing |
| Database & Infrastructure | `resources/database-patterns.md` | Migrations, local Docker infra |
| API Generation | `resources/api-workflows.md` | Generating API clients |
| i18n Patterns | `resources/i18n-patterns.md` | Internationalization |
| Release Coordination | `resources/release-coordination.md` | Versioning, changelog, releases |
| Troubleshooting | `resources/troubleshooting.md` | Debugging issues |

### Task Dependencies

Define dependencies in `mise.toml`:

```toml
[tasks.build]
depends = ["lint", "test"]
run = "echo 'Building after lint and test pass'"

[tasks.dev]
depends = ["//apps/api:dev", "//apps/web:dev"]
```

### Parallel vs Sequential Execution

**Parallel (independent tasks):**
```bash
# Runs all lint tasks simultaneously
mise run lint
```

**Sequential (dependent tasks):**
```bash
# Runs in order: lint → test → build
mise run lint && mise run test && mise run build
```

**Mixed approach:**
```bash
# Start dev servers in background
mise run //apps/api:dev &
mise run //apps/web:dev &
wait
```

### Environment Variables

Common patterns for monorepo env vars:

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/db

# Cache
REDIS_URL=redis://localhost:6379/0

# API
API_URL=http://localhost:8000

# Frontend
PUBLIC_API_URL=http://localhost:8000
```

### Output Templates

When setting up development environment:
1. Runtime installation verification (`mise list`)
2. Dependency installation commands per app
3. Environment variable template (.env.example)
4. Development server startup commands
5. Common task quick reference

When running tasks:
1. Command executed with full path
2. Expected output summary
3. Duration and success/failure status
4. Next recommended actions

When troubleshooting:
1. Diagnostic commands (`mise config`, `mise doctor`)
2. Common issue solutions
3. Port/process conflict resolution
4. Cleanup commands if needed

### Troubleshooting Guide

| Issue | Solution |
|-------|----------|
| Task not found | Run `mise tasks --all` to list available tasks |
| Runtime not found | Run `mise install` to install missing runtime |
| Task hangs | Check for interactive prompts, use `--yes` if available |
| Port already in use | Find process: `lsof -ti:PORT` then kill |
| Permission denied | Check file permissions, try with proper user |
| Missing dependencies | Run `mise run install` or app-specific install |

### How to Execute

Follow the core workflow step by step:
1. **Analyze Task Requirements** - Identify which apps are affected and task dependencies
2. **Check mise Configuration** - Verify mise.toml structure and available tasks
3. **Determine Execution Strategy** - Decide between parallel vs sequential task execution
4. **Run Prerequisites** - Install runtimes, dependencies if needed
5. **Execute Tasks** - Run mise tasks with proper error handling
6. **Verify Results** - Check output, logs, and generated artifacts
7. **Report Status** - Summarize success/failure with actionable next steps

### Execution Protocol (CLI Mode)

Vendor-specific execution protocols are injected automatically by `oma agent:spawn`.
Source files live under `../_shared/runtime/execution-protocols/{vendor}.md`.

## References

- Clarification: `../_shared/core/clarification-protocol.md`
- Difficulty assessment: `../_shared/core/difficulty-guide.md`

### Knowledge Reference

mise, task runner, monorepo, dev server, lint, format, test, typecheck, build, deployment, ci/cd, parallel execution, workflow, automation, tooling
