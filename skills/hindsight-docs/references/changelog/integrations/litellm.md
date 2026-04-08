---
hide_table_of_contents: true
---

import PageHero from '@site/src/components/PageHero';

<PageHero title="LiteLLM Changelog" subtitle="hindsight-litellm — universal LLM memory integration via LiteLLM." />

← LiteLLM integration

## [0.5.0](https://github.com/vectorize-io/hindsight/tree/integrations/litellm/v0.5.0)

**Features**

- Add streaming support when using the LiteLLM wrapper integration. ([`665877bb`](https://github.com/vectorize-io/hindsight/commit/665877bb))
- Add async retain and reflect support, along with a cleaned-up LiteLLM integration API. ([`1d4879a2`](https://github.com/vectorize-io/hindsight/commit/1d4879a2))
- Initial release of the Hindsight LiteLLM integration implementation. ([`dfccbf29`](https://github.com/vectorize-io/hindsight/commit/dfccbf29))

**Improvements**

- Support sending tags and mission metadata through the LiteLLM integration to improve memory organization and retrieval. ([`f3c5a9c1`](https://github.com/vectorize-io/hindsight/commit/f3c5a9c1))

**Bug Fixes**

- When no explicit Hindsight query is provided, the integration now uses the most recent user message as the query to avoid missing/empty memory lookups. ([`5e8952c5`](https://github.com/vectorize-io/hindsight/commit/5e8952c5))
- Fix API key handling by passing the configured api_key through to the Hindsight client in the LiteLLM integration. ([`c0ca9b02`](https://github.com/vectorize-io/hindsight/commit/c0ca9b02))
