---
hide_table_of_contents: true
---

import PageHero from '@site/src/components/PageHero';

<PageHero title="LangGraph Changelog" subtitle="hindsight-langgraph — LangGraph and LangChain memory integration." />

← LangGraph integration

## [0.1.2](https://github.com/vectorize-io/hindsight/tree/integrations/langgraph/v0.1.2)

**Improvements**

- Improved Python typing support for the Hindsight LangGraph integration (added PEP 561 py.typed marker) so type checkers work correctly. ([`d054b884`](https://github.com/vectorize-io/hindsight/commit/d054b884))
- Updated dependencies to address critical/high security vulnerabilities in the Hindsight LangGraph integration. ([`ee4510a7`](https://github.com/vectorize-io/hindsight/commit/ee4510a7))

**Bug Fixes**

- All HTTP requests from the Hindsight LangGraph integration now include a consistent identifying User-Agent header. ([`9372462e`](https://github.com/vectorize-io/hindsight/commit/9372462e))

## [0.1.1](https://github.com/vectorize-io/hindsight/tree/integrations/langgraph/v0.1.1)

**Features**

- Added LangGraph integration for Hindsight. ([`b4320254`](https://github.com/vectorize-io/hindsight/commit/b4320254))
