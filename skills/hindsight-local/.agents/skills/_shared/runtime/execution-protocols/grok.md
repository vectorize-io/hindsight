# Execution Protocol (Grok)

When running as a Grok subagent or custom agent, follow this protocol for shared state coordination with oh-my-agent.

## State Management

Use file-based I/O for coordination. Coordination/state files (task-board, progress,
result hand-offs) MUST be written to the **project-root memory store** `.serena/memories/`
— that is the only location the orchestrator (`oma agent:status`), `oma verify`, and the
memory/retro tooling read. Writing them anywhere else (e.g. `.agents/results/`) leaves them
orphaned and your run is reported as `crashed`. Human-facing deliverables (plans, bug
reports, design docs) belong under `.agents/results/` instead.

Grok has good native support for project files and can use terminal commands for coordination.

### Path Resolution (CRITICAL)

All result, progress, and state files MUST be written to the **project root** `.serena/memories/` directory.

- **Project root** = the git repository root (where `.git` exists)
- **Session-scoped naming**: when running under an orchestration session, append session ID as suffix:
  - `result-{agent-id}-{sessionId}.md`
- **Manual runs**: `result-{agent-id}.md`

## On Start

1. Read the assigned task context (from orchestration or user prompt; see `.serena/memories/task-board.md` if present).
2. Create progress tracking if needed under `.serena/memories/`.

## During Execution

- Use clear, structured output.
- Prefer editing existing files over creating new documentation unless requested.
- Periodically update `progress-{agent-id}[-{sessionId}].md` with current state.

## On Completion

- Create `result-{agent-id}[-{sessionId}].md` with final result including:
  - A status line — see **Status line format** below (REQUIRED)
  - Summary of work done
  - Files created/modified
  - Acceptance criteria checklist

## On Failure

- Still create `result-{agent-id}[-{sessionId}].md` with the status line set to `failed`
- Include detailed error description and what remains incomplete

## Status line format (REQUIRED)

The orchestrator parses the status with the regex `^## Status:\s*(\S+)`. The result file
MUST contain a single line in exactly this shape — heading marker, colon on the same line,
plain word, no backticks, no quotes:

```
## Status: completed
```

Use `## Status: failed` on failure. Do NOT split it across lines or render it as a
sub-bullet (e.g. `- Status: completed`) — that fails to parse and a failed run would be
silently misreported as completed.

## Grok-Specific Notes

- Leverage Grok's strong code understanding and search capabilities.
- Use `run_terminal_cmd` for shell operations.
- Subagent spawning via the `task` tool when appropriate.
- Follow any `agents_md` or project instructions loaded in the agent definition.