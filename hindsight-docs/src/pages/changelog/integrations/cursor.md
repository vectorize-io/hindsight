---
hide_table_of_contents: true
---

import PageHero from '@site/src/components/PageHero';

<PageHero title="Cursor Changelog" subtitle="hindsight-memory - Hindsight memory plugin for Cursor." />

[← Cursor integration](/sdks/integrations/cursor)

## [0.1.0](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/cursor)

**Features**

- Added a Hindsight memory plugin for Cursor using Cursor's plugin hooks for automatic recall and retain.
- Added Cursor-native extras including an always-on rule and an on-demand `hindsight-recall` skill.
- Added support for local daemon mode, external Hindsight APIs, and native MCP as an alternative integration path.
- Added automated tests covering config loading, bank derivation, content formatting, and hook behavior.
