---
hide_table_of_contents: true
---

# OpenCode Integration Changelog

Changelog for [`@vectorize-io/opencode-hindsight`](https://www.npmjs.com/package/@vectorize-io/opencode-hindsight).

For the source code, see [`hindsight-integrations/opencode`](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/opencode).

← [Back to main changelog](../index.md)

## [0.2.0](https://github.com/vectorize-io/hindsight/tree/integrations/opencode/v0.2.0)

**Features**

- Share a single memory bank across multiple Git worktrees in the same repository.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/isac322" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@isac322</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/78e48e59" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>78e48e59</a>

**Improvements**

- Reduce default memory retention frequency to save less often (default retainEveryNTurns: 3 instead of 10).<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/DK09876" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@DK09876</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/902704df" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>902704df</a>

**Bug Fixes**

- Prevent garbled bank IDs by preserving raw UTF-8 when deriving the memory bank identifier dynamically.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/Desko77" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@Desko77</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/08a75b5b" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>08a75b5b</a>

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
