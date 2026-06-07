---
hide_table_of_contents: true
---

import PageHero from '@site/src/components/PageHero';

<PageHero title="LangGraph Changelog" subtitle="hindsight-langgraph — LangGraph and LangChain memory integration." />

← LangGraph integration

## [0.2.0](https://github.com/vectorize-io/hindsight/tree/integrations/langgraph/v0.2.0)

**Breaking Changes**

- Updated the LangGraph integration API by removing the legacy BaseStore-based store and adjusting nodes/tools to use the new memory instructions flow.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/DK09876" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@DK09876</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/b67e813a" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>b67e813a</a>

**Improvements**

- Improved LangGraph integration reliability by fixing graph nodes and adding expanded end-to-end and flow tests.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/DK09876" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@DK09876</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/b67e813a" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>b67e813a</a>

## [0.1.2](https://github.com/vectorize-io/hindsight/tree/integrations/langgraph/v0.1.2)

**Improvements**

- Improved Python typing support for the Hindsight LangGraph integration (added PEP 561 py.typed marker) so type checkers work correctly. ([`d054b884`](https://github.com/vectorize-io/hindsight/commit/d054b884))
- Updated dependencies to address critical/high security vulnerabilities in the Hindsight LangGraph integration. ([`ee4510a7`](https://github.com/vectorize-io/hindsight/commit/ee4510a7))

**Bug Fixes**

- All HTTP requests from the Hindsight LangGraph integration now include a consistent identifying User-Agent header. ([`9372462e`](https://github.com/vectorize-io/hindsight/commit/9372462e))

## [0.1.1](https://github.com/vectorize-io/hindsight/tree/integrations/langgraph/v0.1.1)

**Features**

- Added LangGraph integration for Hindsight. ([`b4320254`](https://github.com/vectorize-io/hindsight/commit/b4320254))
