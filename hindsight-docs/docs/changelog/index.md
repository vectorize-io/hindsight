---
sidebar_position: 1
---

# Changelog

This changelog highlights user-facing changes only. Internal maintenance, CI/CD, and infrastructure updates are omitted.

For full release details, see [GitHub Releases](https://github.com/vectorize-io/hindsight/releases).

## [Unreleased]

**Features**

- Add per-request token usage tracking to retain and reflect endpoints for cost monitoring and billing integration.

## [0.2.0](https://github.com/vectorize-io/hindsight/releases/tag/v0.2.0)

**Features**

- Add additional model provider support, including Anthropic Claude and LM Studio. ([`787ed60`](https://github.com/vectorize-io/hindsight/commit/787ed60))
- Add multi-bank access and new MCP tools for interacting with multiple memory banks via MCP. ([`6b5f593`](https://github.com/vectorize-io/hindsight/commit/6b5f593))
- Allow supplying custom entities when retaining memories via the retain endpoint. ([`dd59bc8`](https://github.com/vectorize-io/hindsight/commit/dd59bc8))
- Enhance the /reflect endpoint with max_tokens control and optional structured output responses. ([`d49e820`](https://github.com/vectorize-io/hindsight/commit/d49e820))


**Improvements**

- Improve local LLM support for reasoning-capable models and streamline Docker startup for local deployments. ([`eea0f27`](https://github.com/vectorize-io/hindsight/commit/eea0f27))
- Support operation validator extensions and return proper HTTP errors when validation fails. ([`ce45d30`](https://github.com/vectorize-io/hindsight/commit/ce45d30))
- Add configurable observation thresholds to control when observations are created/updated. ([`54e2df0`](https://github.com/vectorize-io/hindsight/commit/54e2df0))
- Improve graph visualization to the control plane for exploring memory relationships. ([`1a62069`](https://github.com/vectorize-io/hindsight/commit/1a62069))

**Bug Fixes**

- Fix MCP server lifecycle handling so MCP lifespan is correctly tied to the FastAPI app lifespan. ([`6b78f7d`](https://github.com/vectorize-io/hindsight/commit/6b78f7d))

## [0.1.15](https://github.com/vectorize-io/hindsight/releases/tag/v0.1.15)

**Features**

- Add the ability to delete documents from the web UI. ([`f7ff32d`](https://github.com/vectorize-io/hindsight/commit/f7ff32d))

**Improvements**

- Improve the API health check endpoint and update the generated client APIs/types accordingly. ([`e06a612`](https://github.com/vectorize-io/hindsight/commit/e06a612))

## [0.1.14](https://github.com/vectorize-io/hindsight/releases/tag/v0.1.14)

**Bug Fixes**

- Fixes the embedded “get-skill” installer so installing skills works correctly. ([`0b352d1`](https://github.com/vectorize-io/hindsight/commit/0b352d1))

## [0.1.13](https://github.com/vectorize-io/hindsight/releases/tag/v0.1.13)

**Improvements**

- Improve reliability by surfacing task handler failures so retries can occur when processing fails. ([`904ea4d`](https://github.com/vectorize-io/hindsight/commit/904ea4d))
- Revamp the hindsight-embed component architecture, including a new daemon/client model and CLI updates for embedding workflows. ([`e6511e7`](https://github.com/vectorize-io/hindsight/commit/e6511e7))

**Bug Fixes**

- Fix memory retention so timestamps are correctly taken into account. ([`234d426`](https://github.com/vectorize-io/hindsight/commit/234d426))

## [0.1.12](https://github.com/vectorize-io/hindsight/releases/tag/v0.1.12)

**Features**

- Added an extensions system for plugging in new operations/skills (including built-in tenant support). ([`2a0c490`](https://github.com/vectorize-io/hindsight/commit/2a0c490))
- Introduced the hindsight-embed tool and a native agentic skill for embedding/agent workflows. ([`da44a5e`](https://github.com/vectorize-io/hindsight/commit/da44a5e))

**Improvements**

- Improved reliability when parsing LLM JSON by retrying on parse errors and adding clearer diagnostics. ([`a831a7b`](https://github.com/vectorize-io/hindsight/commit/a831a7b))

**Bug Fixes**

- Fixed structured-output support for Ollama-based LLM providers. ([`32bca12`](https://github.com/vectorize-io/hindsight/commit/32bca12))
- Adjusted LLM validation to cap max completion tokens at 100 to prevent validation failures. ([`b94b5cf`](https://github.com/vectorize-io/hindsight/commit/b94b5cf))

## [0.1.11](https://github.com/vectorize-io/hindsight/releases/tag/v0.1.11)

**Bug Fixes**

- Fixed the standalone Docker image and control plane standalone build process so standalone deployments build correctly. ([`2948cb6`](https://github.com/vectorize-io/hindsight/commit/2948cb6))

## [0.1.10](https://github.com/vectorize-io/hindsight/releases/tag/v0.1.10)

*This release contains internal maintenance and infrastructure changes only.*


## [0.1.9](https://github.com/vectorize-io/hindsight/releases/tag/v0.1.9)

**Features**

- Simplified local MCP installation and added a standalone UI option for easier setup. ([`1c6acc3`](https://github.com/vectorize-io/hindsight/commit/1c6acc3))

**Bug Fixes**

- Fixed the standalone Docker image so it builds and starts reliably. ([`b52eb90`](https://github.com/vectorize-io/hindsight/commit/b52eb90))
- Improved Docker runtime reliability by adding required system utilities (procps). ([`ae80876`](https://github.com/vectorize-io/hindsight/commit/ae80876))

## [0.1.8](https://github.com/vectorize-io/hindsight/releases/tag/v0.1.8)

**Bug Fixes**

- Fix bank list responses when a bank has no name. ([`04f01ab`](https://github.com/vectorize-io/hindsight/commit/04f01ab))
- Fix failures when retaining memories asynchronously. ([`63f5138`](https://github.com/vectorize-io/hindsight/commit/63f5138))
- Fix a race condition in the bank selector when switching banks. ([`e468a4e`](https://github.com/vectorize-io/hindsight/commit/e468a4e))

## [0.1.7](https://github.com/vectorize-io/hindsight/releases/tag/v0.1.7)

*This release contains internal maintenance and infrastructure changes only.*

## [0.1.6](https://github.com/vectorize-io/hindsight/releases/tag/v0.1.6)

**Features**

- Added support for the Gemini 3 Pro and GPT-5.2 models. ([`bb1f9cb`](https://github.com/vectorize-io/hindsight/commit/bb1f9cb))
- Added a local MCP server option for running/connecting to Hindsight via MCP without a separate remote service. ([`7dd6853`](https://github.com/vectorize-io/hindsight/commit/7dd6853))

**Improvements**

- Updated the Postgres/pg0 dependency to a newer 0.11.x series for improved compatibility and stability. ([`47be07f`](https://github.com/vectorize-io/hindsight/commit/47be07f))

## [0.1.5](https://github.com/vectorize-io/hindsight/releases/tag/v0.1.5)

**Features**

- Added LiteLLM integration so Hindsight can capture and manage memories from LiteLLM-based LLM calls. ([`dfccbf2`](https://github.com/vectorize-io/hindsight/commit/dfccbf2))
- Added an optional graph-based retriever (MPFP) to improve recall by leveraging relationships between memories. ([`7445cef`](https://github.com/vectorize-io/hindsight/commit/7445cef))

**Improvements**

- Switched the embedded Postgres layer to pg0-embedded for a smoother local/standalone experience. ([`94c2b85`](https://github.com/vectorize-io/hindsight/commit/94c2b85))

**Bug Fixes**

- Fixed repeated retries on 400 errors from the LLM, preventing unnecessary request loops and failures. ([`70983f5`](https://github.com/vectorize-io/hindsight/commit/70983f5))
- Fixed recall trace visualization in the control plane so search/recall debugging displays correctly. ([`922164e`](https://github.com/vectorize-io/hindsight/commit/922164e))
- Fixed the CLI installer to make installation more reliable. ([`158a6aa`](https://github.com/vectorize-io/hindsight/commit/158a6aa))
- Updated Next.js to patch security vulnerabilities (CVE-2025-55184, CVE-2025-55183). ([`f018cc5`](https://github.com/vectorize-io/hindsight/commit/f018cc5))

## [0.1.3](https://github.com/vectorize-io/hindsight/releases/tag/v0.1.3)

**Improvements**

- Improved CLI and UI branding/polish, including new banner/logo assets and updated interface styling. ([`fa554b8`](https://github.com/vectorize-io/hindsight/commit/fa554b8))


## [0.1.2](https://github.com/vectorize-io/hindsight/releases/tag/v0.1.2)

**Bug Fixes**

- Fixed the standalone Docker image so it builds/runs correctly. ([`1056a20`](https://github.com/vectorize-io/hindsight/commit/1056a20))
