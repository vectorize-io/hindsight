# Changelog

## 0.1.0

Initial release. Port of the [`hindsight-integrations/claude-code`](../claude-code/) plugin to Devin CLI, unblocked by CLI 3000.3.22 exposing `session_id` on every hook event.

- Auto-recall on `UserPromptSubmit`, auto-retain on `Stop`/`SessionEnd`, `agent_knowledge_*` MCP tools, and a `create-agent` skill — same feature set and config schema (`HINDSIGHT_*` env vars, `~/.hindsight/devin-cli.json`) as the Claude Code plugin.
- Retain reads Devin CLI's local session SQLite database (keyed by `session_id`) instead of a JSONL transcript file, since Devin CLI hooks don't expose a `transcript_path`. See `scripts/lib/devin_transcript.py` for details and caveats — this is reading an undocumented, unstable-by-definition CLI internal.
- Hooks and the MCP server are registered by `scripts/install.py` (surfaced as the `setup` skill) directly into `~/.config/devin/config.json` / `~/.config/devin/mcp_config.json`, rather than through Devin CLI's native plugin hooks.json/mcp_config.json auto-wiring — that beta mechanism doesn't yet provide a way for a plugin's own hook/MCP commands to locate their bundled scripts portably. See `scripts/install.py`'s docstring.
