---
hide_table_of_contents: true
---

import PageHero from '@site/src/components/PageHero';

<PageHero title="Claude Code Changelog" subtitle="hindsight-memory — Hindsight memory plugin for Claude Code." />

[← Claude Code integration](/sdks/integrations/claude-code)

## [0.3.0](https://github.com/vectorize-io/hindsight/tree/integrations/claude-code/v0.3.0)

**Features**

- Claude Code integration now retains tool calls as structured JSON for more accurate memory and retrieval. ([`8cb8b912`](https://github.com/vectorize-io/hindsight/commit/8cb8b912))

## [0.2.0](https://github.com/vectorize-io/hindsight/tree/integrations/claude-code/v0.2.0)

**Features**

- Added a Claude Code integration plugin for capturing and using Hindsight memory in Claude Code. ([`f4390bdc`](https://github.com/vectorize-io/hindsight/commit/f4390bdc))
- Claude Code integration can retain full sessions with document upsert and configurable tagging. ([`2d31b67d`](https://github.com/vectorize-io/hindsight/commit/2d31b67d))

**Improvements**

- Improved Claude Code plugin installation and configuration experience. ([`35b2cbb6`](https://github.com/vectorize-io/hindsight/commit/35b2cbb6))
- Integrations no longer rely on hardcoded default models, allowing model selection to be fully configured. ([`58e68f3e`](https://github.com/vectorize-io/hindsight/commit/58e68f3e))
- Claude Code now starts the Hindsight background daemon automatically at session start for smoother operation. ([`26944e25`](https://github.com/vectorize-io/hindsight/commit/26944e25))

**Bug Fixes**

- Added a supported setup command to register hooks reliably, fixing hook registration issues. ([`22ca6a8d`](https://github.com/vectorize-io/hindsight/commit/22ca6a8d))
- Fixed Claude Code integration compatibility on Windows. ([`a94a90ea`](https://github.com/vectorize-io/hindsight/commit/a94a90ea))
