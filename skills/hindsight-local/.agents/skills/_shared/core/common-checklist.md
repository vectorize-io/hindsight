# Common Code Quality Checklist

Apply these checks to ALL code before submitting, regardless of domain.

## Code Quality
- [ ] No hardcoded secrets (API keys, passwords, tokens)
- [ ] No `TODO`/`FIXME` left unresolved
- [ ] Meaningful variable and function names
- [ ] Functions < 50 lines, files < 500 lines
- [ ] Cyclomatic complexity < 10 per function
- [ ] No deeply nested code (< 4 levels)

## Error Handling
- [ ] All async operations have try/catch or error boundaries
- [ ] User-facing error messages are clear and actionable
- [ ] No silent failures (errors logged or surfaced)

## Security
- [ ] No user input directly in SQL/shell/HTML
- [ ] Authentication checked on protected endpoints
- [ ] Sensitive data not in logs or error messages

## Testing
- [ ] Unit tests for new business logic
- [ ] Edge cases covered (empty, null, boundary values)
- [ ] Tests actually assert meaningful behavior

## Git Hygiene
- [ ] Commit message describes the "why", not the "what"
- [ ] No unrelated changes bundled
- [ ] No generated files or secrets committed
