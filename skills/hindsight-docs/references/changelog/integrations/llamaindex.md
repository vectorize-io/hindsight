---
hide_table_of_contents: true
---

# LlamaIndex Integration Changelog

Changelog for [`hindsight-llamaindex`](https://pypi.org/project/hindsight-llamaindex/).

For the source code, see [`hindsight-integrations/llamaindex`](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/llamaindex).

← [Back to main changelog](../index.md)

## [0.1.5](https://github.com/vectorize-io/hindsight/tree/integrations/llamaindex/v0.1.5)

**Improvements**

- Replaced the deprecated manual test with a gated end-to-end test suite, improving integration reliability without requiring a real LLM for every run.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/DK09876" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@DK09876</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/ed34756c" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>ed34756c</a>

**Bug Fixes**

- LlamaIndex integration now defaults to using Hindsight Cloud, improving out-of-the-box connectivity and reducing setup issues.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/DK09876" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}>@DK09876</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/ed34756c" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>ed34756c</a>

## [0.1.4](https://github.com/vectorize-io/hindsight/tree/integrations/llamaindex/v0.1.4)

**Improvements**

- All HTTP requests now include a consistent identifying User-Agent header. ([`9372462e`](https://github.com/vectorize-io/hindsight/commit/9372462e))
- Improved Python type-checking support by shipping a PEP 561 marker file in the package. ([`d054b884`](https://github.com/vectorize-io/hindsight/commit/d054b884))
- Updated dependencies to address critical and high-severity security vulnerabilities. ([`ee4510a7`](https://github.com/vectorize-io/hindsight/commit/ee4510a7))

## [0.1.3](https://github.com/vectorize-io/hindsight/tree/integrations/llamaindex/v0.1.3)

**Bug Fixes**

- Fixed LlamaIndex integration issues with document IDs, the memory API, and ReAct trace handling to improve reliability and correctness. ([`d93dfea8`](https://github.com/vectorize-io/hindsight/commit/d93dfea8))

## [0.1.2](https://github.com/vectorize-io/hindsight/tree/integrations/llamaindex/v0.1.2)

**Features**

- Added LlamaIndex integration for Hindsight. ([`2d787c4f`](https://github.com/vectorize-io/hindsight/commit/2d787c4f))
