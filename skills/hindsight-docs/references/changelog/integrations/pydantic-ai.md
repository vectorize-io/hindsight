---
hide_table_of_contents: true
---

import PageHero from '@site/src/components/PageHero';

<PageHero title="Pydantic AI Changelog" subtitle="hindsight-pydantic-ai — persistent memory tools for Pydantic AI agents." />

← Pydantic AI integration

## [0.4.20](https://github.com/vectorize-io/hindsight/tree/integrations/pydantic-ai/v0.4.20)

**Features**

- Added Pydantic AI integration to provide persistent agent memory in Hindsight. ([`cab5a40f`](https://github.com/vectorize-io/hindsight/commit/cab5a40f))

**Improvements**

- Set an identifying User-Agent header on all HTTP requests made by clients. ([`9372462e`](https://github.com/vectorize-io/hindsight/commit/9372462e))
- Added a PEP 561 py.typed marker so type checkers recognize bundled type hints. ([`d054b884`](https://github.com/vectorize-io/hindsight/commit/d054b884))
- Updated dependencies to address critical and high-severity security vulnerabilities. ([`ee4510a7`](https://github.com/vectorize-io/hindsight/commit/ee4510a7))
