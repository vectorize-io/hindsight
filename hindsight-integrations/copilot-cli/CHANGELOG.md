# Changelog

## [0.1.1] - 2026-07-30

### Fixed

- Parse GitHub Copilot CLI's native session transcript format (Copilot CLI
  >= 1.0.76). Copilot now writes transcript events with dotted names
  (`user.message`, `assistant.message`) and the message text under `data`,
  which the parser did not recognize — it extracted zero messages, so
  auto-retain silently stored nothing on newer Copilot builds. The parser now
  reads the native envelope (using the clean `data.content`, ignoring the
  reminder-injected `transformedContent`), with a regression test pinning the
  1.0.76 shape.

## [0.1.0]

### Added

- Initial release. Persistent long-term memory for GitHub Copilot CLI via
  hooks: recall on `sessionStart` (and `subagentStart`), retain on `agentStop`
  (every `retainEveryNTurns`) and a forced final retain on `sessionEnd`.
  Per-repo bank scoping derived from the working directory, configurable recall
  budget and retain frequency, and user- or repo-scoped hook registration.
