# Backend Agent - Examples

## Example 1: CRUD API Endpoint

**Input**: "Create a TODO CRUD API with user ownership"

**Output**:
```
## Task: TODO CRUD API

### Endpoints Implemented
- POST /api/todos (create)
- GET /api/todos (list, user-scoped)
- GET /api/todos/{id} (detail)
- PATCH /api/todos/{id} (update)
- DELETE /api/todos/{id} (delete)

### Database Schema
- todos table: id (UUID), title (VARCHAR), completed (BOOL), user_id (FK), created_at
- Index on user_id

### Security
- [x] All endpoints require JWT auth
- [x] User can only access own todos (user_id filter)
- [x] Input validated with schema validation

### Files Created
- app/models/todo.*
- app/schemas/todo.*
- app/services/todo_service.*
- app/api/todos.*
- tests/test_todos.*
- migrations/xxx_add_todos.*
```

## Example 2: Authentication System

**Input**: "Add JWT authentication with register and login"

**Output**:
```
## Task: JWT Authentication

### Endpoints Implemented
- POST /api/auth/register (email + password)
- POST /api/auth/login (returns access + refresh tokens)
- POST /api/auth/refresh (refresh token rotation)

### Security
- [x] Password hashed with bcrypt (cost 12)
- [x] Access token: 15min expiry
- [x] Refresh token: 7 day expiry, single-use rotation
- [x] Rate limiting: 5 attempts/min on login
- [x] Email validation enforced

### Files Created
- app/models/user.*
- app/schemas/auth.*
- app/services/auth_service.*
- app/api/auth.*
- app/core/security.* (JWT helpers, password hashing)
- tests/test_auth.*
```

## Example 3: Fixing N+1 Query

**Input**: "The /api/posts endpoint is slow, taking 2s for 50 posts"

**Output**:
```
## Task: Fix N+1 Query in Posts API

### Root Cause
Each post triggers a separate query for its author (N+1 problem).

### Fix
Added eager loading for the Post.author relation.

### Before: 51 queries (1 + 50)
### After: 1 query with JOIN

### Files Modified
- app/api/posts.* (added eager loading)
- tests/test_posts.* (added performance assertion)
```
