# API Contracts

This directory is the **SSOT for the contract format only** — it holds `template.md`
and this README. **Do not write generated contracts here** (that pollutes the skill
SSOT tree with run artifacts).

Generated contracts live in one of two places, by lifecycle:

| Contract kind | Location | Tracked by git? |
|---------------|----------|-----------------|
| Run artifact (transient coordination state for a single plan/orchestrate run) | `.agents/results/api-contracts/{domain}.md` | No (gitignored) |
| Durable spec (versioned module/API boundary shipped with a feature) | `docs/plans/contracts/{domain}.md` | Yes |

## Usage

### PM Agent (Author)
Design the contract using `template.md`, then write the generated contract to the
run-artifact location during the planning phase:
```
[WRITE](".agents/results/api-contracts/{domain}.md", contract content)
```
If the contract must be versioned as a durable spec, promote it to
`docs/plans/contracts/{domain}.md` when committing the feature.

If MCP memory tool is unavailable, create files directly at the locations above.

### Backend Agent (Implementer)
Read contract and implement exactly as specified:
```
[READ](".agents/results/api-contracts/{domain}.md")   # or docs/plans/contracts/{domain}.md
```

### Frontend / Mobile Agent (Consumer)
Read contract and integrate API client exactly as specified:
```
[READ](".agents/results/api-contracts/{domain}.md")   # or docs/plans/contracts/{domain}.md
```

## Tool Reference

Tool names are configured in `mcp.json → memoryConfig.tools`:
- `[READ]` → default: `read_memory`
- `[WRITE]` → default: `write_memory`

## Contract Format

```markdown
# {Domain} API Contract

## POST /api/{resource}
- **Auth**: Required (JWT Bearer)
- **Request Body**:
  ```json
  { "field": "type", "field2": "type" }
  ```
- **Response 200**:
  ```json
  { "id": "uuid", "field": "value", "created_at": "ISO8601" }
  ```
- **Response 401**: `{ "detail": "Not authenticated" }`
- **Response 422**: `{ "detail": [{ "field": "error message" }] }`
```

## When to Create

- **New API endpoint**: PM Agent creates contract before implementation tasks are assigned
- **Existing API schema change**: Update contract first, then notify affected agents
- **Cross-platform feature**: Contract must exist before backend/frontend/mobile tasks start

## Completion Criteria

- [ ] Request schema defined with all required/optional fields
- [ ] Response schema defined (200, 201, etc.)
- [ ] Error cases documented (400, 401, 403, 404, 422, 500)
- [ ] Authentication requirements specified
- [ ] Rate limiting noted (if applicable)
- [ ] Backend Agent has reviewed and approved
- [ ] Frontend/Mobile Agent has reviewed and approved

## Rules

1. PM Agent must create during planning
2. Backend Agent must not implement differently from contract
3. Frontend/Mobile Agent defines types based on contract
4. If changes are needed, request re-planning from PM Agent
