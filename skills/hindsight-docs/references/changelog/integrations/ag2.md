---
hide_table_of_contents: true
---

# ag2 Integration Changelog

Changelog for [`hindsight-ag2`](https://pypi.org/project/hindsight-ag2/).

For the source code, see [`hindsight-integrations/ag2`](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/ag2).

← [Back to main changelog](../index.md)

## [0.1.2](https://github.com/vectorize-io/hindsight/tree/integrations/ag2/v0.1.2)

**Improvements**

- Improved type-checking support for the AG2 integration by shipping PEP 561 type information. ([`d054b884`](https://github.com/vectorize-io/hindsight/commit/d054b884))

**Bug Fixes**

- All HTTP requests from the AG2 integration now include an identifying User-Agent header for improved compatibility and observability. ([`9372462e`](https://github.com/vectorize-io/hindsight/commit/9372462e))
- Resolved critical and high-severity security vulnerabilities in dependencies. ([`ee4510a7`](https://github.com/vectorize-io/hindsight/commit/ee4510a7))

## [0.1.1](https://github.com/vectorize-io/hindsight/tree/integrations/ag2/v0.1.1)

**Features**

- Added AG2 framework integration for Hindsight. ([`73123870`](https://github.com/vectorize-io/hindsight/commit/73123870))
