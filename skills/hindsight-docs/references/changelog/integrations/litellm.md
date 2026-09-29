---
hide_table_of_contents: true
---

import PageHero from '@site/src/components/PageHero';

<PageHero title="LiteLLM Changelog" subtitle="hindsight-litellm — universal LLM memory integration via LiteLLM." />

[← LiteLLM integration](../../sdks/integrations/litellm.md)

## [0.5.4](https://github.com/vectorize-io/hindsight/tree/integrations/litellm/v0.5.4)

**Bug Fixes**

- Fixes LiteLLM integration reliability by correcting injection mode behavior, ensuring context-manager state is restored properly, and making validation/errors consistent.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/DK09876" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@DK09876</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/dfe74b1d" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>dfe74b1d</a>

## [0.5.3](https://github.com/vectorize-io/hindsight/tree/integrations/litellm/v0.5.3)

*This release contains internal maintenance and infrastructure changes only.*

## [0.5.2](https://github.com/vectorize-io/hindsight/tree/integrations/litellm/v0.5.2)

**Bug Fixes**

- Fix conversation storage when using streaming LiteLLM responses.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/DK09876" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@DK09876</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/ac5181f5" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>ac5181f5</a>

## [0.5.1](https://github.com/vectorize-io/hindsight/tree/integrations/litellm/v0.5.1)

**Improvements**

- Add bundled type information for better type-checking support when using the integration in typed Python projects. ([`d054b884`](https://github.com/vectorize-io/hindsight/commit/d054b884))
- Improve security and stability by updating and constraining the LiteLLM dependency, including excluding a compromised version. ([`8a2388a4`](https://github.com/vectorize-io/hindsight/commit/8a2388a4))

**Bug Fixes**

- Set an identifying User-Agent header on all HTTP requests to improve compatibility with providers and proxies. ([`9372462e`](https://github.com/vectorize-io/hindsight/commit/9372462e))

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
