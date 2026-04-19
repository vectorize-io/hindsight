---
hide_table_of_contents: true
---

import PageHero from '@site/src/components/PageHero';

<PageHero title="OpenAI Codex CLI Changelog" subtitle="Hindsight memory integration for OpenAI Codex CLI." />

[← Codex CLI integration](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/codex)

## [0.2.1](https://github.com/vectorize-io/hindsight/tree/integrations/codex/v0.2.1)

**Improvements**

- All Codex integration HTTP requests now include an identifying User-Agent header for better request tracking and compatibility. ([`9372462e`](https://github.com/vectorize-io/hindsight/commit/9372462e))

## [0.2.0](https://github.com/vectorize-io/hindsight/tree/integrations/codex/v0.2.0)

**Features**

- Retain structured Codex tool calls from rollout files so they’re preserved in Hindsight memory. ([`3461398b`](https://github.com/vectorize-io/hindsight/commit/3461398b))

## [0.1.1](https://github.com/vectorize-io/hindsight/tree/integrations/codex/v0.1.1)

**Features**

- Added Hindsight memory integration for the OpenAI Codex CLI, enabling Codex to use and store memories in Hindsight. ([`0b17a67c`](https://github.com/vectorize-io/hindsight/commit/0b17a67c))

## [0.1.0](https://github.com/vectorize-io/hindsight/tree/integrations/codex/v0.1.0)

**Features**

- Added Hindsight memory integration for OpenAI Codex CLI with three hook scripts: SessionStart (daemon warm-up), UserPromptSubmit (auto-recall), and Stop (auto-retain). ([`0b17a67c`](https://github.com/vectorize-io/hindsight/commit/0b17a67c))
- Full-session retain with session-level upsert using session ID as document ID. ([`0b17a67c`](https://github.com/vectorize-io/hindsight/commit/0b17a67c))
- Dynamic bank IDs for per-project memory isolation. ([`0b17a67c`](https://github.com/vectorize-io/hindsight/commit/0b17a67c))
- Automatic daemon lifecycle management with background pre-start. ([`0b17a67c`](https://github.com/vectorize-io/hindsight/commit/0b17a67c))
- 57 automated tests covering content processing and end-to-end hook behavior. ([`71125cd9`](https://github.com/vectorize-io/hindsight/commit/71125cd9))
