---
hide_table_of_contents: true
---

# Strands Integration Changelog

Changelog for [`hindsight-strands`](https://pypi.org/project/hindsight-strands/).

For the source code, see [`hindsight-integrations/strands`](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/strands).

← [Back to main changelog](/changelog)

## [0.1.3](https://github.com/vectorize-io/hindsight/tree/integrations/strands/v0.1.3)

**Bug Fixes**

- Fixes Strands integration to properly close internally owned Hindsight clients, preventing resource leaks and related stability issues.<span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/benfrank241" target="_blank" rel="noopener noreferrer" style={{color: "var(--ifm-color-primary)", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px", verticalAlign: "middle"}}><img src="https://github.com/benfrank241.png?size=40" alt="@benfrank241" width="18" height="18" style={{borderRadius: "50%"}} />@benfrank241</a><span style={{color: "var(--ifm-color-emphasis-500)", margin: "0 0.3em"}}>·</span><a href="https://github.com/vectorize-io/hindsight/commit/2bfd7747" target="_blank" rel="noopener noreferrer" style={{fontFamily: "var(--ifm-font-family-monospace, monospace)", fontSize: "0.85em", color: "var(--ifm-color-emphasis-600)"}}>2bfd7747</a>

## [0.1.2](https://github.com/vectorize-io/hindsight/tree/integrations/strands/v0.1.2)

**Improvements**

- Improved Python typing support for the Strands integration by shipping the PEP 561 "py.typed" marker. ([`d054b884`](https://github.com/vectorize-io/hindsight/commit/d054b884))

**Bug Fixes**

- All Strands integration HTTP requests now include a consistent identifying User-Agent for better compatibility and troubleshooting. ([`9372462e`](https://github.com/vectorize-io/hindsight/commit/9372462e))

## [0.1.1](https://github.com/vectorize-io/hindsight/tree/integrations/strands/v0.1.1)

**Features**

- Added Strands Agents SDK integration, enabling Hindsight memory tools to be used with Strands agents. ([`7fe773c0`](https://github.com/vectorize-io/hindsight/commit/7fe773c0))
