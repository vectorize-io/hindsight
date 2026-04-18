# Changelog

## [0.4.0] - 2026-04-18

### Changed (breaking)
- **Dynamic `bank_id` values are no longer URL-encoded at construction time.**
  Previously, each granularity segment (project, agent, session, etc.) was
  passed through `urllib.parse.quote(..., safe="")` before being joined with
  `::`. This caused bank names to be stored server-side with percent-encoded
  characters for any non-ASCII project folder (e.g. `мой проект` was stored
  as `%D0%BC%D0%BE%D0%B9%20%D0%BF%D1%80%D0%BE%D0%B5%D0%BA%D1%82`), which made
  banks unreadable in the UI and CLI.
  bank_id is now stored as raw UTF-8. HTTP path encoding still happens in the
  client transport layer, which is correct.

### Upgrade notes
- After upgrading, users on `dynamicBankId: true` will start writing to a new
  bank whose name reflects the raw project folder. Existing percent-encoded
  banks remain on the server untouched and can be renamed or left as archive.

## [0.1.0] - 2025-03-23

### Added
- Initial release: Claude Code plugin for Hindsight long-term memory
- Auto-recall on every user prompt via `UserPromptSubmit` hook — injects relevant memories as `additionalContext`
- Auto-retain after every response via async `Stop` hook — extracts and stores conversation transcript
- Session lifecycle hooks (`SessionStart` health check, `SessionEnd` daemon cleanup)
- Three connection modes: external API, auto-managed local daemon (`uvx hindsight-embed`), existing local server
- Dynamic bank IDs with configurable granularity (`agent`, `project`, `session`, `channel`, `user`)
- Channel-agnostic: works with Claude Code Channels (Telegram, Discord, Slack) and interactive sessions
- Zero pip dependencies — pure Python stdlib (`urllib`, `fcntl`, `subprocess`)
- 34 configuration options via `settings.json` with env var overrides
- LLM auto-detection from `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`
- Chunked retention with sliding window (`retainEveryNTurns` + `retainOverlapTurns`)
- Memory tag stripping to prevent retain feedback loops
