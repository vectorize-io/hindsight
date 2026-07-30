# Changelog

## [Unreleased]

### Added

- `profile` setting (env: `HINDSIGHT_PROFILE`) with a `coding` preset that
  tunes grouped defaults for coding sessions: `retainToolCalls: true`,
  per-project dynamic banks (`dynamicBankId` + `["agent", "project"]`),
  `recallBudget: "low"`, and engineering-focused bank/retain missions.
  Presets override built-in and vendor-shipped defaults but never a key set
  explicitly in user config or env, so existing configurations are
  unaffected; without a profile, behavior is unchanged.
- Retained documents are automatically tagged with `file:<relpath>` for each
  file modified by `Write`/`Edit`/`MultiEdit`/`NotebookEdit` tool calls
  (relativized against the working directory, capped at 20), and the list is
  recorded in `metadata.files_modified`. Recall can then filter with
  `recallTags: ["file:src/auth.py"]`.

- `{user_id}` template variable for `retainTags` and `retainMetadata`, resolved
  from the `HINDSIGHT_USER_ID` env var (empty string if unset). Enables
  machine-independent per-user memory scoping without hardcoding user ids in
  `settings.json`.
- `requestTimeoutSeconds` config (env: `HINDSIGHT_REQUEST_TIMEOUT_SECONDS`) to
  override the per-call HTTP timeout used by recall (10s), retain (15s) and the
  knowledge MCP tools. Defaults to `null`, which preserves current per-call
  behavior. Set this when self-hosted Hindsight legitimately takes longer than
  10s under contention (e.g. parallel recalls) so the client doesn't surface
  `read operation timed out` on requests the server completes successfully.
  Does not affect the health check, which stays at 5s. Fixes #1575.

### Fixed

- `tool_use` inputs are now size-capped before retention (300 chars per
  string field, priority-keys-only fallback above 1500 serialized chars).
  Previously a `Write` of a large file embedded the entire file body in the
  retain payload, bloating requests and degrading fact extraction —
  `tool_result` blocks were already truncated at 2000 chars but inputs were
  not.

### Changed

- Tags that resolve to an empty namespace content (e.g. `"user:"` when
  `HINDSIGHT_USER_ID` is unset) are now dropped from retain requests. Previously
  such tags were sent as-is. Tags without `:` are unaffected.

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
