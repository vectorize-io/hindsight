---
hide_table_of_contents: true
---

import PageHero from '@site/src/components/PageHero';

<PageHero title="Cursor Changelog" subtitle="hindsight-memory - Hindsight memory plugin for Cursor." />

[← Cursor integration](/sdks/integrations/cursor)

## [0.2.0](https://github.com/vectorize-io/hindsight/tree/integrations/cursor/v0.2.0)

**Features**

- Added a Hindsight memory plugin integration for Cursor.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/DK09876" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}><img src="https://github.com/DK09876.png?size=40" alt="@DK09876" width="18" height="18" style={{borderRadius: "50%"}} />@DK09876</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/91d767cdc" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>91d767cdc</a>

## [0.1.0](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/cursor)

**Features**

- Added a Hindsight memory plugin for Cursor using Cursor's plugin hooks for automatic recall and retain.
- Added Cursor-native extras including an always-on rule and an on-demand `hindsight-recall` skill.
- Added support for local daemon mode, external Hindsight APIs, and native MCP as an alternative integration path.
- Added automated tests covering config loading, bank derivation, content formatting, and hook behavior.
