# PM Agent - Examples

## Example 1: Simple TODO App

**Input**: "Build a TODO app with JWT authentication"

**Output**:
```json
{
  "project_name": "TODO App with JWT Auth",
  "tech_stack": "Resolve from each domain skill's resources/tech-stack.md at planning time",
  "tasks": [
    {
      "id": "task-1",
      "agent": "backend",
      "title": "JWT authentication API",
      "priority": 1,
      "dependencies": [],
      "estimated_complexity": "high",
      "acceptance_criteria": [
        "POST /api/auth/register with email + password",
        "POST /api/auth/login returns access + refresh tokens",
        "Password hashed with bcrypt",
        "Rate limiting: 5/min on login"
      ]
    },
    {
      "id": "task-2",
      "agent": "backend",
      "title": "TODO CRUD API",
      "priority": 1,
      "dependencies": [],
      "acceptance_criteria": [
        "CRUD endpoints for /api/todos",
        "User-scoped (JWT required)",
        "Pagination on list endpoint"
      ]
    },
    {
      "id": "task-3",
      "agent": "frontend",
      "title": "Login + Register UI",
      "priority": 1,
      "dependencies": [],
      "acceptance_criteria": [
        "Login and register forms with validation",
        "JWT token storage",
        "Redirect to /todos after login"
      ]
    },
    {
      "id": "task-4",
      "agent": "frontend",
      "title": "TODO List UI",
      "priority": 2,
      "dependencies": ["task-2", "task-3"],
      "acceptance_criteria": [
        "Add, toggle, delete todos",
        "Loading and empty states",
        "Responsive design"
      ]
    },
    {
      "id": "task-5",
      "agent": "qa",
      "title": "Security & Performance Review",
      "priority": 3,
      "dependencies": ["task-1", "task-2", "task-3", "task-4"],
      "acceptance_criteria": [
        "OWASP Top 10 audit passed",
        "Lighthouse > 90",
        "No N+1 queries"
      ]
    }
  ]
}
```

## Example 2: Feature Addition to Existing App

**Input**: "Add a comment system to the blog posts"

**Output**:
```json
{
  "project_name": "Blog Comments Feature",
  "tasks": [
    {
      "id": "task-1",
      "agent": "backend",
      "title": "Comments API",
      "priority": 1,
      "dependencies": [],
      "acceptance_criteria": [
        "POST /api/posts/{id}/comments (auth required)",
        "GET /api/posts/{id}/comments (public, paginated)",
        "DELETE /api/comments/{id} (owner only)",
        "Nested replies (1 level deep)"
      ]
    },
    {
      "id": "task-2",
      "agent": "frontend",
      "title": "Comment Section UI",
      "priority": 2,
      "dependencies": ["task-1"],
      "acceptance_criteria": [
        "Comment list with pagination (load more)",
        "Add comment form (auth required)",
        "Reply to comment",
        "Delete own comment",
        "Real-time count update"
      ]
    }
  ]
}
```

## Example 3: Standards-Aligned Delivery Plan

**Input**: "Plan this enterprise release with risk and governance considerations"

**Output**:
```json
{
  "project_name": "Enterprise Release Plan",
  "architecture_decisions": [
    {
      "decision": "Use phased rollout with feature flags",
      "rationale": "Reduces operational risk during release",
      "alternatives_considered": ["big bang release", "tenant-by-tenant rollout"]
    }
  ],
  "project_controls": {
    "iso_21500": {
      "scope_defined": true,
      "stakeholders_identified": ["product", "security", "operations"],
      "dependencies_mapped": true
    },
    "iso_31000": {
      "top_risks": [
        "migration rollback failure",
        "auth regression on legacy users"
      ],
      "treatments": [
        "pre-release restore drill",
        "shadow auth validation"
      ]
    },
    "iso_38500": {
      "decision_owner": "engineering manager",
      "approval_required_for": ["prod migration", "feature-flag enablement"]
    }
  }
}
```
