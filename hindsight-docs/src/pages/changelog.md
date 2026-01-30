---
hide_table_of_contents: true
---

# Changelog

This changelog highlights user-facing changes only. Internal maintenance, CI/CD, and infrastructure updates are omitted.

For full release details, see [GitHub Releases](https://github.com/vectorize-io/hindsight/releases).

## [0.4.0](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.0)

**Observations**, **Mental Models**, new **Agentic Reflect** and Directives, read the [announcement](/blog/learning-capabilities).

**Features**

- Added support for providing a custom prompt for memory extraction. ([`3172e99`](https://github.com/vectorize-io/hindsight/commit/3172e99))
- Expanded the LiteLLM integration with async retain/reflect support, cleaner API, and support for tags/mission (including passing API keys correctly). ([`1d4879a`](https://github.com/vectorize-io/hindsight/commit/1d4879a))
- Added a new worker service to run background tasks at scale. ([`4c79240`](https://github.com/vectorize-io/hindsight/commit/4c79240))
- MCP retain now supports timestamps. ([`b378f68`](https://github.com/vectorize-io/hindsight/commit/b378f68))
- Added support for installing skills via `npx add-skill`. ([`ec22317`](https://github.com/vectorize-io/hindsight/commit/ec22317))

**Improvements**

- CLI retain-files now accepts more file types. ([`1eeced3`](https://github.com/vectorize-io/hindsight/commit/1eeced3))

**Bug Fixes**

- Fixed a macOS crash in the embed daemon caused by an XPC connection issue. ([`e5fc6ee`](https://github.com/vectorize-io/hindsight/commit/e5fc6ee))
- Fixed occasional extraction in the wrong language. ([`87d4a36`](https://github.com/vectorize-io/hindsight/commit/87d4a36))
- Fixed PyTorch model initialization issues that could cause startup failures (meta tensor/init problems). ([`ddaa5f5`](https://github.com/vectorize-io/hindsight/commit/ddaa5f5))


**Features**

- Add memory tags so you can label and filter memories during recall/reflect. ([`20c8f8b`](https://github.com/vectorize-io/hindsight/commit/20c8f8b))
- Allow choosing different AI providers/models per operation. ([`e6709d5`](https://github.com/vectorize-io/hindsight/commit/e6709d5))
- Add Cohere support for embeddings and reranking. ([`4de0730`](https://github.com/vectorize-io/hindsight/commit/4de0730))
- Add configurable embedding dimensions and OpenAI embeddings support. ([`70de23e`](https://github.com/vectorize-io/hindsight/commit/70de23e))
- Support custom base URLs for OpenAI-style embeddings and Cohere endpoints. ([`fa53917`](https://github.com/vectorize-io/hindsight/commit/fa53917))
- Add LiteLLM gateway support for routing LLM/embedding requests. ([`d47c8a2`](https://github.com/vectorize-io/hindsight/commit/d47c8a2))
- Add multilingual content support to improve handling and retrieval across languages. ([`c65c6a9`](https://github.com/vectorize-io/hindsight/commit/c65c6a9))
- Add delete memory bank capability. ([`4b82d2d`](https://github.com/vectorize-io/hindsight/commit/4b82d2d))
- Add backup/restore tooling for memory banks. ([`67b273d`](https://github.com/vectorize-io/hindsight/commit/67b273d))

**Improvements**

- Add retention modes to control how memories are extracted and stored. ([`fb31a35`](https://github.com/vectorize-io/hindsight/commit/fb31a35))
- Add offline (optional) database migrations to support restricted/air-gapped deployments. ([`233bd2e`](https://github.com/vectorize-io/hindsight/commit/233bd2e))
- Add database connection configuration options for more flexible deployments. ([`33fac2c`](https://github.com/vectorize-io/hindsight/commit/33fac2c))
- Load .env automatically on startup to simplify configuration. ([`c06d9b4`](https://github.com/vectorize-io/hindsight/commit/c06d9b4))
- Expose an operation ID from retain requests so async/background processing can be tracked. ([`1dacd0e`](https://github.com/vectorize-io/hindsight/commit/1dacd0e))
- Add per-request LLM token usage metrics for monitoring and cost tracking. ([`29a542d`](https://github.com/vectorize-io/hindsight/commit/29a542d))
- Add LLM call latency metrics for performance monitoring. ([`5e1f13e`](https://github.com/vectorize-io/hindsight/commit/5e1f13e))
- Include tenant in metrics labels for better multi-tenant observability. ([`1ffc2a4`](https://github.com/vectorize-io/hindsight/commit/1ffc2a4))
- Add async processing option to MCP retain tool for background retention workflows. ([`37fc7fb`](https://github.com/vectorize-io/hindsight/commit/37fc7fb))

**Bug Fixes**

- Fix extension loading in multi-worker deployments so all workers load extensions correctly. ([`f5f3fca`](https://github.com/vectorize-io/hindsight/commit/f5f3fca))
- Improve recall performance by batching recall queries. ([`5991308`](https://github.com/vectorize-io/hindsight/commit/5991308))
- Improve retrieval quality and stability for large memory banks (graph/MPFP retrieval fixes). ([`6232e69`](https://github.com/vectorize-io/hindsight/commit/6232e69))
- Fix entities list being limited to 100 entities. ([`26bf571`](https://github.com/vectorize-io/hindsight/commit/26bf571))
- Fix UI only showing the first 1000 memories. ([`67c1a42`](https://github.com/vectorize-io/hindsight/commit/67c1a42))
- Fix duplicated causal relationships and improve token usage during processing. ([`49e233c`](https://github.com/vectorize-io/hindsight/commit/49e233c))
- Improve causal link detection accuracy. ([`2a00df0`](https://github.com/vectorize-io/hindsight/commit/2a00df0))
- Make retain max completion tokens configurable to prevent truncation issues. ([`7715a51`](https://github.com/vectorize-io/hindsight/commit/7715a51))
- Fix Python SDK not sending the Authorization header, preventing authenticated requests. ([`39e3f7c`](https://github.com/vectorize-io/hindsight/commit/39e3f7c))
- Fix stats endpoint missing tenant authentication in multi-tenant setups. ([`d6ff191`](https://github.com/vectorize-io/hindsight/commit/d6ff191))
- Fix embedding dimension handling for tenant schemas in multi-tenant databases. ([`6fe9314`](https://github.com/vectorize-io/hindsight/commit/6fe9314))
- Fix Groq free-tier compatibility so requests work correctly. ([`d899d18`](https://github.com/vectorize-io/hindsight/commit/d899d18))
- Fix security vulnerability (qs / CVE-2025-15284). ([`b3becb6`](https://github.com/vectorize-io/hindsight/commit/b3becb6))
- Restore MCP tools for listing and creating memory banks. ([`9fd5679`](https://github.com/vectorize-io/hindsight/commit/9fd5679))

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
