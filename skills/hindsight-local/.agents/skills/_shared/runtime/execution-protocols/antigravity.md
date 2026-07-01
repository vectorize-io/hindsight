# Execution Protocol (Antigravity)

When running as a CLI subagent (`agy -p` headless mode), follow this protocol for shared
state coordination. **In headless mode your stdout is discarded by the spawner** — the only
durable hand-off to the orchestrator is the result artifact written below. If you do not
write it, the orchestrator reports your run as `crashed` even on success.

## MCP Memory Tools

Tool names are configurable via `mcp_config.json → memoryConfig.tools`:
- `[READ]` → default: `read_memory`
- `[WRITE]` → default: `write_memory`
- `[EDIT]` → default: `edit_memory`
- `[LIST]` → default: `list_memories`
- `[DELETE]` → default: `delete_memory`

Memory base path is configurable via `memoryConfig.basePath` (default: `.serena/memories`).

If Serena MCP memory tools are unavailable, fall back to writing the same files directly to
`.serena/memories/` using your native file-write tool.

### Path Resolution (CRITICAL)

All result, progress, and state files MUST be written to the **project root** memory path, never to a subdirectory's memory path.

- **Session-scoped naming**: when running under an orchestration session, append session ID as suffix:
  - `result-{agent-id}-{sessionId}.md` (e.g., `result-frontend-session-20260405-100835.md`)
  - `progress-{agent-id}-{sessionId}.md`
- **Manual (non-orchestrated) runs**: no suffix, `result-{agent-id}.md`

## On Start

1. `[READ]("task-board.md")` to confirm your assigned task
2. `[WRITE]("progress-{agent-id}[-{sessionId}].md", initial progress entry)` with Turn 1 status

## During Execution

- Every 3-5 turns: `[EDIT]("progress-{agent-id}[-{sessionId}].md")` to append a new turn entry
- Include: action taken, current status, files created/modified

## On Completion

- `[WRITE]("result-{agent-id}[-{sessionId}].md")` with final result including:
  - A status line — see **Status line format** below (REQUIRED)
  - Summary of work done
  - Files created/modified
  - Acceptance criteria checklist

## On Failure

- Still create `result-{agent-id}[-{sessionId}].md` with the status line set to `failed`
- Include detailed error description and what remains incomplete

## Status line format (REQUIRED)

The orchestrator parses the status with the regex `^## Status:\s*(\S+)`. The result file
MUST contain a single line in exactly this shape — colon on the same line, plain word, no
backticks, no quotes:

```
## Status: completed
```

Use `## Status: failed` on failure. Do NOT split it across lines (e.g. `## Status` then a
separate ``` `completed` ``` line) — that fails to parse and a failed run would be silently
misreported as completed.
