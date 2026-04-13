---
hide_table_of_contents: true
---

import PageHero from '@site/src/components/PageHero';

<PageHero title="OpenClaw Changelog" subtitle="@vectorize-io/hindsight-openclaw — Hindsight memory plugin for OpenClaw." />

← OpenClaw integration

## [0.6.1](https://github.com/vectorize-io/hindsight/tree/integrations/openclaw/v0.6.1)

**Bug Fixes**

- Openclaw setup wizard now prompts for the actual token value instead of an environment variable name. ([`9679d813`](https://github.com/vectorize-io/hindsight/commit/9679d813))

## 0.6.1 (Unreleased)

**Improvements**

- `hindsight-openclaw-setup` interactive wizard now asks for the API token/API key **value** instead of the env var name holding it. Pasted values are masked and stored inline in `openclaw.json` — no more two-step "pick an env var name, then export it" flow that confused first-time users. For CI / production, the SecretRef path is still available via the non-interactive flags (`--token-env`, `--api-key-env`) or after-the-fact with `openclaw config set ... --ref-source env --ref-id …`.
- Added non-interactive CLI flags for direct-value credentials: `--token` (cloud / external API modes) and `--api-key` (embedded mode). `--token` and `--token-env` are mutually exclusive within a mode; same for `--api-key` / `--api-key-env`.

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

## 0.6.0 (Unreleased)

**Breaking Changes**

- The plugin no longer reads any configuration from process environment variables. All settings — including the LLM provider, model, API key, base URL, external Hindsight API URL/token, and bank ID — must now be set through OpenClaw's plugin config (e.g. `openclaw config set plugins.entries.hindsight-openclaw.config.<field> <value>`). API keys and other secrets should be configured as `SecretRef` values via `--ref-source env|file|exec` so they're resolved from your secret store at runtime instead of being stored in plaintext on disk.
- Removed the `llmApiKeyEnv` plugin config field. Use the new `llmApiKey` field configured as a SecretRef instead (e.g. `openclaw config set plugins.entries.hindsight-openclaw.config.llmApiKey --ref-source env --ref-id OPENAI_API_KEY`).
- Removed automatic LLM provider detection from `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` / `GROQ_API_KEY`. Set `llmProvider` and `llmApiKey` explicitly via `openclaw config set`.
- Removed support for the `HINDSIGHT_API_LLM_PROVIDER`, `HINDSIGHT_API_LLM_MODEL`, `HINDSIGHT_API_LLM_API_KEY`, `HINDSIGHT_API_LLM_BASE_URL`, `HINDSIGHT_EMBED_API_URL`, `HINDSIGHT_EMBED_API_TOKEN`, and `HINDSIGHT_BANK_ID` environment variables. The same values now live in plugin config — see the migration guide.

**Features**

- Added `hindsight-openclaw-setup`, an interactive setup wizard that walks users through picking one of three install modes — **Cloud** (managed Hindsight at `https://api.hindsight.vectorize.io`), **External API** (your own running Hindsight deployment), or **Embedded daemon** (local `hindsight-embed` daemon). The wizard writes a valid plugin config with env-backed `SecretRef` credentials and no plaintext secrets on disk.
- `hindsight-openclaw-setup` also runs non-interactively via `--mode cloud|api|embedded` plus mode-specific flags (`--api-url`, `--token-env`, `--no-token`, `--provider`, `--api-key-env`, `--model`) for CI and scripted installs.
- Added the `llmApiKey` plugin config field, marked as a sensitive field so OpenClaw resolves it as a `SecretRef` from env, file, or exec sources.
- Added the `llmBaseUrl` plugin config field for OpenAI-compatible endpoint overrides (OpenRouter, Azure OpenAI, vLLM, etc.).
- Marked `hindsightApiToken` as a sensitive field — it can now be configured as a `SecretRef` the same way as `llmApiKey`.

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
