---
hide_table_of_contents: true
---

import PageHero from '@site/src/components/PageHero';

<PageHero title="OpenClaw Changelog" subtitle="@vectorize-io/hindsight-openclaw — Hindsight memory plugin for OpenClaw." />

← OpenClaw integration

## [0.6.5](https://github.com/vectorize-io/hindsight/tree/integrations/openclaw/v0.6.5)

**Bug Fixes**

- Fix per-session memory retention so saving a new turn no longer overwrites earlier turns in the same session.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/nicoloboschi" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@nicoloboschi</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/1f897314" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>1f897314</a>
- Improve direct-execution detection by correctly resolving full symlink chains, preventing incorrect behavior when invoked through symlinked paths.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/D2758695161" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@D2758695161</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/7ceaa22a" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>7ceaa22a</a>

## [0.6.4](https://github.com/vectorize-io/hindsight/tree/integrations/openclaw/v0.6.4)

**Features**

- Openclaw now uses a session-scoped document ID and records structured timestamps for each message, improving session consistency and message ordering. ([`33645e08`](https://github.com/vectorize-io/hindsight/commit/33645e08))

## [0.6.3](https://github.com/vectorize-io/hindsight/tree/integrations/openclaw/v0.6.3)

**Features**

- Inline retain tags are now merged with default retain tags, improving consistency of retention behavior without extra configuration. ([`b79ab2b7`](https://github.com/vectorize-io/hindsight/commit/b79ab2b7))

**Improvements**

- All HTTP requests now include an identifying User-Agent, improving request attribution and compatibility with stricter endpoints. ([`9372462e`](https://github.com/vectorize-io/hindsight/commit/9372462e))

**Bug Fixes**

- Per-agent banking now respects identity skip-filter configuration, preventing incorrect identity filtering. ([`90a22016`](https://github.com/vectorize-io/hindsight/commit/90a22016))

## [0.6.2](https://github.com/vectorize-io/hindsight/tree/integrations/openclaw/v0.6.2)

**Features**

- OpenClaw conversations are now stored in an Anthropic-style JSON format, preserving tool_use/tool_result blocks for more faithful replay and analysis. ([`adc85129`](https://github.com/vectorize-io/hindsight/commit/adc85129))

**Bug Fixes**

- Improved session consistency and reduced noise by stabilizing session identity and skipping non-user operational turns. ([`2ff805d6`](https://github.com/vectorize-io/hindsight/commit/2ff805d6))
- Fixed intermittent missing behavior by ensuring agent hooks are registered on every plugin invocation. ([`1be5ff33`](https://github.com/vectorize-io/hindsight/commit/1be5ff33))

## [0.6.1](https://github.com/vectorize-io/hindsight/tree/integrations/openclaw/v0.6.1)

**Bug Fixes**

- Openclaw setup wizard now prompts for the actual token value instead of an environment variable name. ([`9679d813`](https://github.com/vectorize-io/hindsight/commit/9679d813))

## [0.6.0](https://github.com/vectorize-io/hindsight/tree/integrations/openclaw/v0.6.0)

**Breaking Changes**

- Configuration is now read from the plugin configuration instead of environment variables, requiring updates to existing deployments. ([`e22ae05f`](https://github.com/vectorize-io/hindsight/commit/e22ae05f))

**Features**

- Adds an interactive setup wizard with Cloud, API, and Embedded configuration modes. ([`87322396`](https://github.com/vectorize-io/hindsight/commit/87322396))
- Adds a daemon lifecycle package for running the Hindsight "all" daemon. ([`576016f5`](https://github.com/vectorize-io/hindsight/commit/576016f5))
- Adds a configuration-aware CLI to backfill historical data into Hindsight memory. ([`72fd3d59`](https://github.com/vectorize-io/hindsight/commit/72fd3d59))
- Adds session pattern filtering to ignore or treat certain sessions as stateless. ([`5a61ac50`](https://github.com/vectorize-io/hindsight/commit/5a61ac50))
- Adds configurable tags for retained memories. ([`b0e8ac0f`](https://github.com/vectorize-io/hindsight/commit/b0e8ac0f))
- Adds support for bankId when using static banks. ([`0e81d1a2`](https://github.com/vectorize-io/hindsight/commit/0e81d1a2))

**Improvements**

- Improves startup resilience and enriches retained memory metadata. ([`1f1716bd`](https://github.com/vectorize-io/hindsight/commit/1f1716bd))
- Adds a JSONL-backed retain queue to improve reliability when the external API is unavailable. ([`087545cc`](https://github.com/vectorize-io/hindsight/commit/087545cc))
- Reduces CLI startup time by deferring heavy initialization until the service starts. ([`41025c3b`](https://github.com/vectorize-io/hindsight/commit/41025c3b))

**Bug Fixes**

- Avoids misrouting by ignoring ctx.channelId when it contains a provider name. ([`d4b8b354`](https://github.com/vectorize-io/hindsight/commit/d4b8b354))

## [0.5.1](https://github.com/vectorize-io/hindsight/tree/integrations/openclaw/v0.5.1)

**Bug Fixes**

- Fixed JSON manifest formatting issues in the OpenClaw plugin to prevent manifest parsing/loading problems. ([`704e41fa`](https://github.com/vectorize-io/hindsight/commit/704e41fa))

## [0.5.0](https://github.com/vectorize-io/hindsight/tree/integrations/openclaw/v0.5.0)

**Breaking Changes**

- Removed hardcoded default model settings from integrations so model/provider must be configured explicitly. ([`58e68f3e`](https://github.com/vectorize-io/hindsight/commit/58e68f3e))

**Features**

- Added configurable, structured logging for the OpenClaw integration. ([`d441ab81`](https://github.com/vectorize-io/hindsight/commit/d441ab81))
- Added an auto-recall toggle and support for excluding specific providers from recall/retention. ([`3f9eb27c`](https://github.com/vectorize-io/hindsight/commit/3f9eb27c))
- Added configuration to skip recall/retention for selected providers. ([`fb7be3ec`](https://github.com/vectorize-io/hindsight/commit/fb7be3ec))
- Added dynamic per-channel memory banks to isolate memory across channels. ([`9a776e9f`](https://github.com/vectorize-io/hindsight/commit/9a776e9f))
- Added support for using an external Hindsight API backend. ([`6b346925`](https://github.com/vectorize-io/hindsight/commit/6b346925))
- Added plugin configuration options to select the LLM provider and model. ([`8564135b`](https://github.com/vectorize-io/hindsight/commit/8564135b))

**Improvements**

- Added control over where recalled memories are injected to better preserve prompt caching. ([`200bab23`](https://github.com/vectorize-io/hindsight/commit/200bab23))
- Improved recall/retention controls and scalability, and added Gemini safety settings support. ([`d425e93c`](https://github.com/vectorize-io/hindsight/commit/d425e93c))
- Memory retention now periodically keeps recent conversation turns (default every 10 turns) to improve continuity. ([`ad1660b3`](https://github.com/vectorize-io/hindsight/commit/ad1660b3))
- Improved OpenClaw and embedding parameters for better integration behavior and configuration. ([`749478d9`](https://github.com/vectorize-io/hindsight/commit/749478d9))
- Improved OpenClaw configuration setup and initialization behavior. ([`27498f99`](https://github.com/vectorize-io/hindsight/commit/27498f99))

**Bug Fixes**

- Added a configurable auto-recall timeout to prevent recalls from hanging or taking too long. ([`cd4d449f`](https://github.com/vectorize-io/hindsight/commit/cd4d449f))
- Recalled memories are now injected as system context for more reliable behavior. ([`b17f338e`](https://github.com/vectorize-io/hindsight/commit/b17f338e))
- Health check requests now include the auth token to avoid unauthorized failures. ([`40b02645`](https://github.com/vectorize-io/hindsight/commit/40b02645))
- Improved stability and safety with better shell handling, HTTP mode support, lazy reinitialization, and per-user memory banks. ([`c4610130`](https://github.com/vectorize-io/hindsight/commit/c4610130))
- Fixed failures when ingesting very large content (E2BIG). ([`6bad6673`](https://github.com/vectorize-io/hindsight/commit/6bad6673))
- Prevented memory retention from recursing indefinitely. ([`4f112101`](https://github.com/vectorize-io/hindsight/commit/4f112101))
- Prevented user memories from being wiped on every new session. ([`981cf605`](https://github.com/vectorize-io/hindsight/commit/981cf605))
- Improved shell argument escaping to prevent command failures with special characters. ([`63e2964a`](https://github.com/vectorize-io/hindsight/commit/63e2964a))
- Renamed the OpenClaw binary to the correct name to avoid invocation/config mismatches. ([`b364bc34`](https://github.com/vectorize-io/hindsight/commit/b364bc34))
