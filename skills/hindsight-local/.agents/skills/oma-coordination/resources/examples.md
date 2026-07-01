# Workflow Guide - Examples

## Example 1: Full-Stack TODO App

**Input**: "Build a TODO app with JWT authentication"

**Workflow**:

```
Step 1: PM Agent plans the project
  -> 5 tasks: auth API, CRUD API, login UI, todo UI, QA review

Step 2: Spawn Priority 1 agents via CLI
  # Run in parallel using background processes
  oma agent:spawn backend "JWT authentication API + TODO CRUD" session-id -w ./backend &
  oma agent:spawn frontend "Login/Register UI" session-id -w ./frontend &
  wait

Step 3: Monitor progress
  - Use memory read tool to poll progress-{agent}.md files
  - Verify API contracts align between backend/frontend

Step 4: Spawn Priority 2 after P1 completes
  oma agent:spawn frontend "TODO List UI" session-id -w ./frontend

Step 5: Spawn Priority 3
  oma agent:spawn qa "Security + Performance review" session-id -w .

Step 6: Address QA findings
  - Re-spawn agents for CRITICAL/HIGH issues
  - Verify fixes
```

## Example 2: Adding Feature to Existing App

**Input**: "Add a comment feature to the blog"

**Workflow**:

```
Step 1: PM Agent analyzes existing codebase and plans
  -> 2 tasks: Comments API, Comment Section UI

Step 2: Spawn Backend Agent first (API-first)
  oma agent:spawn backend "Comments API with nested replies, pagination" session-id -w ./backend

Step 3: After backend completes, spawn Frontend Agent
  oma agent:spawn frontend "Comment section UI using the new API endpoints" session-id -w ./frontend
  (Sequential because frontend depends on API contracts)

Step 4: QA review
  oma agent:spawn qa "Security: XSS in comments, rate limiting; Performance: Pagination, N+1 queries" session-id -w .
```

## Example 3: When to Use Orchestrator Instead

**User**: "Run everything automatically"

**Response**:

```
"You want fully automated execution. Let me switch to the
orchestrator skill instead, which will:
1. Spawn CLI subagents automatically
2. Coordinate via Serena Memory
3. Monitor progress and handle retries
4. Return unified results

Use oma-coordination when you want step-by-step control.
Use oma-orchestrator when you want hands-off automation."
```
