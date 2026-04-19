---
hide_table_of_contents: true
---

import PageHero from '@site/src/components/PageHero';

<PageHero title="CrewAI Changelog" subtitle="hindsight-crewai — persistent memory for CrewAI agents." />

← CrewAI integration

## [0.4.20](https://github.com/vectorize-io/hindsight/tree/integrations/crewai/v0.4.20)

**Features**

- Added CrewAI integration enabling persistent memory for crews via Hindsight. ([`41db2960`](https://github.com/vectorize-io/hindsight/commit/41db2960))

**Improvements**

- All HTTP requests now include an identifying User-Agent header for better compatibility and observability with upstream services. ([`9372462e`](https://github.com/vectorize-io/hindsight/commit/9372462e))
- Improved type-checking support for Python users by shipping PEP 561 type information in all packages. ([`d054b884`](https://github.com/vectorize-io/hindsight/commit/d054b884))
- Security updates applied to address known dependency vulnerabilities. ([`300d089b`](https://github.com/vectorize-io/hindsight/commit/300d089b))
