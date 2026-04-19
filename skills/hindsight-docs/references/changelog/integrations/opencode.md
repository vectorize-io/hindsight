---
hide_table_of_contents: true
---

# OpenCode Integration Changelog

Changelog for [`@vectorize-io/opencode-hindsight`](https://www.npmjs.com/package/@vectorize-io/opencode-hindsight).

For the source code, see [`hindsight-integrations/opencode`](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/opencode).

← [Back to main changelog](../index.md)

## [0.1.4](https://github.com/vectorize-io/hindsight/tree/integrations/opencode/v0.1.4)

**Improvements**

- Reduce noisy error output by logging opencode integration errors only in debug mode. ([`33442f19`](https://github.com/vectorize-io/hindsight/commit/33442f19))

## [0.1.3](https://github.com/vectorize-io/hindsight/tree/integrations/opencode/v0.1.3)

**Bug Fixes**

- Fixes the OpenCode integration to correctly parse messages, avoid shared-state issues, and retain content after compaction. ([`6076354a`](https://github.com/vectorize-io/hindsight/commit/6076354a))

## [0.1.2](https://github.com/vectorize-io/hindsight/tree/integrations/opencode/v0.1.2)

**Features**

- Added configuration options to filter recalls by tags (recallTags) and control tag matching behavior (recallTagsMatch). ([`b57e337f`](https://github.com/vectorize-io/hindsight/commit/b57e337f))

**Bug Fixes**

- Fixed the session messages API to return the correct data shape for the OpenCode plugin. ([`fd87de9c`](https://github.com/vectorize-io/hindsight/commit/fd87de9c))
