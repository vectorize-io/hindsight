# OpenCode Integration Changelog

## 0.1.0 (2026-04-03)

Initial release.

- **Auto-retain**: Captures completed OpenCode sessions to Hindsight on `session.idle` events
- **MCP recall**: Documentation for using Hindsight's built-in MCP server with OpenCode for on-demand recall/retain/reflect
- **Compaction memory**: Injects recalled memories into session compaction context
- **Dynamic bank IDs**: Per-project memory isolation based on working directory
- **Daemon management**: Auto-starts `hindsight-embed` locally when no external server is configured
- **Backfill CLI**: Multi-source Python CLI for importing historical sessions from OpenCode SQLite database and JSONL transcript files
- **Configuration**: Full config file support (`~/.hindsight/opencode.json`) with environment variable overrides
