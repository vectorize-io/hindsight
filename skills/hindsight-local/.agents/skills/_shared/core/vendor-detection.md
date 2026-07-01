# Vendor Detection Protocol

When executing a workflow, determine your runtime environment using this priority order.
Then resolve the target vendor for each agent from `.agents/oma-config.yaml` (`agent_cli_mapping`, `default_cli`).

Important:
- Do **not** choose one spawn strategy for the entire workflow based only on the main runtime vendor.
- Decide per agent:
  - `current_runtime_vendor`
  - `target_vendor_for_agent`
  - whether that exact runtime can invoke that target vendor natively
- If native invocation is not available for that agent, fall back to `oma agent:spawn`.

## Detection Order (use first match)

1. **Claude Code**: Your system prompt contains "You are Claude Code" OR the `Agent` tool is available
2. **Codex CLI**: Your system prompt contains "Codex CLI" OR the `apply_patch` tool is available
3. **Gemini CLI**: This file was auto-loaded from `.agents/skills/` AND `@` subagent syntax is available
4. **Antigravity IDE**: This file was auto-loaded from `.agents/skills/` AND no `@` subagent syntax
5. **CLI Fallback**: None of the above matched → use `oma agent:spawn`

## Vendor-Specific Spawn Methods

| Vendor | Spawn Method | Result Handling |
|:---|:---|:---|
| Claude Code | `Agent` tool with `.claude/agents/{name}.md` | Synchronous return |
| Codex CLI | Native custom agents in `.codex/agents/{name}.toml` via `codex exec "@agent ..."` when available, otherwise `oma agent:spawn` | JSON output |
| Gemini CLI | `.gemini/agents/{name}.md` native subagents via `gemini -p "@agent ..."` when available, otherwise `oma agent:spawn` | JSON output or MCP memory poll |
| Antigravity | Prefer `oma agent:spawn` unless a native role-subagent path is explicitly verified for the target vendor | MCP memory poll |
| CLI Fallback | `oma agent:spawn {agent} {prompt} {session} -w {workspace}` | Result file poll |

## Dispatch Rule

For each agent:

1. Resolve `target_vendor_for_agent` from config
2. If `target_vendor_for_agent === current_runtime_vendor` and that runtime has a verified native role-subagent path for that vendor, use the vendor variant agent definition
3. Otherwise, use `oma agent:spawn`

Example:
- Runtime: Claude Code
- Mapping: `frontend: claude`, `backend: claude`, `qa: gemini`
- Result:
  - `frontend` -> native Claude subagent
  - `backend` -> native Claude subagent
  - `qa` -> external Gemini spawn
