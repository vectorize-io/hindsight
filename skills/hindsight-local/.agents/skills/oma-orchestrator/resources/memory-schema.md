# MCP Memory Schema for Multi-Agent Orchestration

## Overview

Each subagent writes only to its own dedicated files. The orchestrator manages session-level files. This ownership model prevents write conflicts between concurrent agents.

## Configuration

Memory base path and tool names are configurable via `mcp.json`:
```json
{
  "memoryConfig": {
    "basePath": ".serena/memories",
    "tools": {
      "read": "read_memory",
      "write": "write_memory",
      "edit": "edit_memory"
    }
  }
}
```

Default base path: `.serena/memories`

## File Structure

```
{memoryConfig.basePath}/
  orchestrator-session.md              # Session metadata (orchestrator only)
  task-board.md                        # Master task list (orchestrator writes, agents read)
  progress-{agent-id}-{sessionId}.md  # Per-agent progress log (owning agent only)
  result-{agent-id}-{sessionId}.md    # Per-agent final result (owning agent only)
```

> **Path rule**: All files MUST be at the project root memory path. In monorepos, never write to a subdirectory's memory path. The session ID suffix prevents conflicts between concurrent sessions.

## orchestrator-session.md

Created by the orchestrator at session start. Updated throughout execution.

```markdown
# Orchestrator Session
## ID: session-{YYYYMMDD}-{HHMMSS}
## Started: {ISO timestamp}
## Status: running | completed | failed | aborted

## Agents
| Agent ID | CLI | PID | Status | Task |
|----------|-----|-----|--------|------|
| backend  | gemini | 12345 | running | task-1 |
| frontend | gemini | 12346 | completed | task-2 |
| mobile   | claude | 12347 | running | task-3 |

## Configuration
- MAX_PARALLEL: 3
- MAX_RETRIES: 2
- POLL_INTERVAL: 30s

## Summary (filled on completion)
- Total Tasks: X
- Completed: X
- Failed: X
- Files Created: [list]
- Issues: [list]
```

## task-board.md

Master task list created by the orchestrator. Subagents read this to understand their assignment but never write to it.

```markdown
# Task Board
## Session: session-{YYYYMMDD}-{HHMMSS}

### task-1
- **Agent**: backend
- **CLI**: gemini
- **Title**: JWT authentication API
- **Status**: pending | in_progress | completed | failed | blocked
- **Priority**: 1
- **Dependencies**: none
- **Description**: Implement user registration and login with JWT tokens
- **Acceptance Criteria**:
  - POST /api/auth/register with email + password
  - POST /api/auth/login returns access + refresh tokens
  - Password hashing with bcrypt
  - Rate limiting on auth endpoints

### task-2
- **Agent**: frontend
- **CLI**: gemini
- **Title**: Login and registration UI
- **Status**: pending
- **Priority**: 1
- **Dependencies**: none
- **Description**: Build login and registration forms with validation
- **Acceptance Criteria**:
  - Login form with email + password
  - Registration form with validation
  - Error handling for failed auth
  - JWT token storage and auto-refresh

### task-3
- **Agent**: qa
- **CLI**: gemini
- **Title**: Security and performance review
- **Status**: blocked
- **Priority**: 2
- **Dependencies**: task-1, task-2
- **Description**: Review all deliverables for security and performance
- **Acceptance Criteria**:
  - OWASP Top 10 security check
  - Performance benchmarks pass
  - No critical or high severity issues
```

## progress-{agent-id}-{sessionId}.md

Each agent creates this file at start and appends entries every 3-5 turns. Only the owning agent writes to this file.

```markdown
# Progress: backend
## Task: task-1
## Agent ID: backend
## Started: {ISO timestamp}

### Turn 1 - {ISO timestamp}
- **Action**: Reading task-board.md and understanding requirements
- **Status**: in_progress
- **Details**: Identified 4 acceptance criteria for JWT auth API

### Turn 5 - {ISO timestamp}
- **Action**: Created user model and auth endpoints
- **Status**: in_progress
- **Details**: Implemented POST /api/auth/register and /api/auth/login
- **Files**: app/models/user.py, app/api/auth.py

### Turn 12 - {ISO timestamp}
- **Action**: Added tests and rate limiting
- **Status**: in_progress
- **Details**: 15 test cases passing, rate limiter configured

### Turn 18 - {ISO timestamp}
- **Action**: All acceptance criteria met
- **Status**: completed
- **Summary**: JWT auth API fully implemented with tests
```

## result-{agent-id}-{sessionId}.md

Each agent creates this file upon completion (success or failure). Only the owning agent writes to this file.

```markdown
# Result: backend
## Task: task-1
## Status: completed
## Turns Used: 18

## Summary
Implemented JWT authentication API with user registration, login,
password hashing (bcrypt), and rate limiting.

## Files Created/Modified
- `app/models/user.py` (NEW) - User model with password hashing
- `app/api/auth.py` (NEW) - Auth endpoints
- `app/middleware/rate_limit.py` (NEW) - Rate limiting middleware
- `tests/test_auth.py` (NEW) - 15 test cases
- `alembic/versions/001_create_users.py` (NEW) - Migration

## Acceptance Criteria
- [x] POST /api/auth/register with email + password
- [x] POST /api/auth/login returns access + refresh tokens
- [x] Password hashing with bcrypt
- [x] Rate limiting on auth endpoints

## Issues Encountered
- None

## Notes
- Access token expiry: 15 minutes
- Refresh token expiry: 7 days
- Rate limit: 5 requests/minute on login endpoint
```

### Failed Result Example

```markdown
# Result: backend
## Task: task-1
## Status: failed
## Turns Used: 20

## Summary
Unable to complete JWT auth API. Database connection issues prevented testing.

## Files Created/Modified
- `app/models/user.py` (NEW) - User model (untested)
- `app/api/auth.py` (NEW) - Auth endpoints (untested)

## Acceptance Criteria
- [x] POST /api/auth/register with email + password
- [x] POST /api/auth/login returns access + refresh tokens
- [x] Password hashing with bcrypt
- [ ] Rate limiting on auth endpoints (NOT DONE)

## Issues Encountered
- PostgreSQL connection refused on localhost:5432
- Could not run integration tests
- Rate limiting implementation incomplete due to turn limit

## Notes
- Needs database running to complete testing
- Rate limiting code is written but untested
```
