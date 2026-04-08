---
hide_table_of_contents: true
---

import PageHero from '@site/src/components/PageHero';

<PageHero title="OpenClaw Changelog" subtitle="@vectorize-io/hindsight-openclaw — Hindsight memory plugin for OpenClaw." />

← OpenClaw integration

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
