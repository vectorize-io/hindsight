# Backend Agent - Self-Verification Checklist

Run through every item before submitting your work.

## API Design
- [ ] RESTful conventions followed (proper HTTP methods, status codes)
- [ ] OpenAPI documentation complete (all endpoints documented)
- [ ] Request/response schemas defined with validation library
- [ ] Pagination for list endpoints returning > 20 items
- [ ] Consistent error response format

## Database
- [ ] Migrations created and tested
- [ ] Indexes on foreign keys and frequently queried columns
- [ ] No N+1 queries; relation loading strategy chosen explicitly for the ORM in use
- [ ] No over-fetching; selected only required fields/columns/attributes
- [ ] Transactions used for multi-step operations with explicit unit-of-work boundaries
- [ ] ORM session/client/entity-manager lifecycle matches the framework's concurrency model
- [ ] Query risks reviewed: missing indexes, full scans, repeated identical queries, join row multiplication

## Security
- [ ] JWT authentication on protected endpoints
- [ ] Password hashing with bcrypt (cost 10-12)
- [ ] Rate limiting on auth endpoints
- [ ] Input validation enforced (no raw user input in queries)
- [ ] SQL injection protected (ORM or parameterized queries)
- [ ] No secrets in code or logs

## Testing
- [ ] Unit tests for service layer logic
- [ ] Integration tests for all endpoints (happy + error paths)
- [ ] Auth scenarios tested (missing token, expired, wrong role)
- [ ] Test coverage > 80%

## Code Quality
- [ ] Clean architecture layers: router -> service -> repository
- [ ] No business logic in route handlers
- [ ] Async/await used consistently
- [ ] Type annotations on all function signatures

## Cloud Readiness
- [ ] No hardcoded config values (DB URLs, API keys, ports); all from env vars
- [ ] No in-process state between requests (sessions, caches, counters)
- [ ] Logs written to stdout/stderr, not file; structured format (JSON) preferred
- [ ] Graceful shutdown handled for background jobs and open connections
