---
hide_table_of_contents: true
---

# Changelog

This changelog highlights user-facing changes only. Internal maintenance, CI/CD, and infrastructure updates are omitted.

For full release details, see [GitHub Releases](https://github.com/vectorize-io/hindsight/releases).

## Integration Changelogs

| Integration | Package | Description |
|---|---|---|
| [LiteLLM](./integrations/litellm) | `hindsight-litellm` | Universal LLM memory via LiteLLM (100+ providers) |
| [Pydantic AI](./integrations/pydantic-ai) | `hindsight-pydantic-ai` | Persistent memory tools for Pydantic AI agents |
| [CrewAI](./integrations/crewai) | `hindsight-crewai` | Persistent memory for CrewAI agents |
| [AI SDK](./integrations/ai-sdk) | `@vectorize-io/hindsight-ai-sdk` | Memory integration for Vercel AI SDK |
| [Chat SDK](./integrations/chat) | `@vectorize-io/hindsight-chat` | Memory integration for Vercel Chat SDK |
| [OpenClaw](./integrations/openclaw) | `@vectorize-io/hindsight-openclaw` | Hindsight memory plugin for OpenClaw |

## [0.4.17](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.17)

**Features**

- Added a manual retry option for failed asynchronous operations. ([`dcaacbe4`](https://github.com/vectorize-io/hindsight/commit/dcaacbe4))
- You can now change/update tags on an existing document. ([`1b4ad7f4`](https://github.com/vectorize-io/hindsight/commit/1b4ad7f4))
- Added history tracking and a diff view for mental model changes. ([`e2baca8b`](https://github.com/vectorize-io/hindsight/commit/e2baca8b))
- Added observation history tracking with a UI diff view to review changes over time. ([`576473b6`](https://github.com/vectorize-io/hindsight/commit/576473b6))
- File uploads can now choose a parser per request, with configurable fallback chains. ([`99220d05`](https://github.com/vectorize-io/hindsight/commit/99220d05))
- Added an extension hook that runs after file-to-Markdown conversion completes. ([`1d17dea2`](https://github.com/vectorize-io/hindsight/commit/1d17dea2))

**Improvements**

- Operations view now supports filtering by operation type and has more reliable auto-refresh behavior. ([`f7a60f89`](https://github.com/vectorize-io/hindsight/commit/f7a60f89))
- Added token limits for “source facts” used during consolidation and recall to better control context usage. ([`5d05962d`](https://github.com/vectorize-io/hindsight/commit/5d05962d))
- Improved bank selector usability by truncating very long bank names in the dropdown. ([`1e40cd22`](https://github.com/vectorize-io/hindsight/commit/1e40cd22))

**Bug Fixes**

- Fixed webhook schema issues affecting multi-tenant retain webhooks. ([`32a4882a`](https://github.com/vectorize-io/hindsight/commit/32a4882a))
- Fixed file ingestion failures by stripping null bytes from parsed file content before retaining. ([`cd3a6a22`](https://github.com/vectorize-io/hindsight/commit/cd3a6a22))
- Fixed tool selection handling for OpenAI-compatible providers when using named tool_choice. ([`1cdfb7c2`](https://github.com/vectorize-io/hindsight/commit/1cdfb7c2))
- Improved consolidation behavior to prioritize a bank’s mission over an ephemeral-state heuristic. ([`00ccf0b2`](https://github.com/vectorize-io/hindsight/commit/00ccf0b2))
- Fixed database migrations to correctly handle mental model embedding dimension changes. ([`7accac94`](https://github.com/vectorize-io/hindsight/commit/7accac94))
- Fixed file upload failures caused by an Iris parser httpx read timeout. ([`fa3501d4`](https://github.com/vectorize-io/hindsight/commit/fa3501d4))
- Improved reliability of running migrations by serializing Alembic upgrades within the process. ([`f88b50a4`](https://github.com/vectorize-io/hindsight/commit/f88b50a4))
- Fixed Google Cloud Storage authentication when using Workload Identity Federation credentials. ([`d2504ac5`](https://github.com/vectorize-io/hindsight/commit/d2504ac5))
- Fixed the bank selector to refresh the bank list when the dropdown is opened. ([`0ad8c2d0`](https://github.com/vectorize-io/hindsight/commit/0ad8c2d0))

## [0.4.16](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.16)

**Features**

- Added Webhooks with `consolidation.completed` and `retain.completed` events. ([`abbf874d`](https://github.com/vectorize-io/hindsight/commit/abbf874d))

**Improvements**

- Improved OpenClaw recall/retention controls. ([`d425e93c`](https://github.com/vectorize-io/hindsight/commit/d425e93c))
- Improved search/reranking quality by switching combined scoring to multiplicative boosts. ([`aa8e5475`](https://github.com/vectorize-io/hindsight/commit/aa8e5475))
- Improved performance of observation recall by 40x on large banks. ([`ad2cf72a`](https://github.com/vectorize-io/hindsight/commit/ad2cf72a))
- Improved server shutdown behavior by capping graceful shutdown time and allowing a forced kill on a second Ctrl+C. ([`4c058b4b`](https://github.com/vectorize-io/hindsight/commit/4c058b4b))

**Bug Fixes**

- Fixed an async deadlock risk by running database schema migrations in a background thread during startup. ([`e0a2ac63`](https://github.com/vectorize-io/hindsight/commit/e0a2ac63))
- Fixed webhook delivery/outbox processing so transactions don’t silently roll back due to using the wrong database schema name. ([`75b95106`](https://github.com/vectorize-io/hindsight/commit/75b95106))
- Fixed observation results to correctly resolve and return related chunks using source_memory_ids. ([`cb6d1c46`](https://github.com/vectorize-io/hindsight/commit/cb6d1c46))
- Fixed MCP bank-level tool filtering compatibility with FastMCP 3.x. ([`f17406fd`](https://github.com/vectorize-io/hindsight/commit/f17406fd))
- Fixed crashes when an LLM returns invalid JSON across all retries (now handled cleanly instead of raising a TypeError). ([`66423b85`](https://github.com/vectorize-io/hindsight/commit/66423b85))
- Fixed observations without source dates to preserve missing (None) temporal fields instead of incorrectly populating them. ([`891c33b1`](https://github.com/vectorize-io/hindsight/commit/891c33b1))

## [0.4.15](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.15)

**Features**

- Added observation_scopes to control the granularity/visibility of observations. ([`55af4681`](https://github.com/vectorize-io/hindsight/commit/55af4681))
- List documents API now supports filtering by tags (and fixes the q parameter description). ([`1d70abfe`](https://github.com/vectorize-io/hindsight/commit/1d70abfe))
- Added PydanticAI integration for persistent agent memory. ([`cab5a40f`](https://github.com/vectorize-io/hindsight/commit/cab5a40f))
- Added richer entity label support (optional labels, free-form values, multi-value fields, and UI polish). ([`9b96becc`](https://github.com/vectorize-io/hindsight/commit/9b96becc))
- Added support for timestamp="unset" so content can be retained without a date. ([`f903948a`](https://github.com/vectorize-io/hindsight/commit/f903948a))
- OpenClaw can now automatically retain the last n+2 turns every n turns (default n=10). ([`ad1660b3`](https://github.com/vectorize-io/hindsight/commit/ad1660b3))
- Added configurable Gemini/Vertex AI safety settings for LLM calls. ([`73ef99e7`](https://github.com/vectorize-io/hindsight/commit/73ef99e7))
- Added extension hooks to customize root routing and error headers. ([`e407f4bc`](https://github.com/vectorize-io/hindsight/commit/e407f4bc))

**Improvements**

- Improved recall performance by fetching all recall chunks in a single query. ([`61bf428b`](https://github.com/vectorize-io/hindsight/commit/61bf428b))
- Improved recall/retain performance and scalability for large memory banks. ([`7942f181`](https://github.com/vectorize-io/hindsight/commit/7942f181))

**Bug Fixes**

- Fixed the TypeScript SDK to send null (not undefined) when includeEntities is false. ([`15f4b876`](https://github.com/vectorize-io/hindsight/commit/15f4b876))
- Prevented reflect from failing with context_length_exceeded on large memory banks. ([`77defd96`](https://github.com/vectorize-io/hindsight/commit/77defd96))
- Fixed a consolidation deadlock caused by retrying after zombie processing tasks. ([`c2876490`](https://github.com/vectorize-io/hindsight/commit/c2876490))
- Fixed observations count in the control plane that always showed 0. ([`eaeaa1f2`](https://github.com/vectorize-io/hindsight/commit/eaeaa1f2))
- Fixed ZeroEntropy rerank endpoint URL and ensured the MCP retain async_processing parameter is handled correctly. ([`f6f1a7d8`](https://github.com/vectorize-io/hindsight/commit/f6f1a7d8))
- Fixed JSON serialization issues and logging-related exception propagation when using the claude_code LLM provider. ([`ecb833f4`](https://github.com/vectorize-io/hindsight/commit/ecb833f4))
- Added bank-scoped request validation to prevent cross-bank/invalid bank operations. ([`5270aa5a`](https://github.com/vectorize-io/hindsight/commit/5270aa5a))

## [0.4.14](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.14)

**Features**

- Add Chat SDK integration to give chatbots persistent memory. ([`fed987f9`](https://github.com/vectorize-io/hindsight/commit/fed987f9))
- Allow configuring which MCP tools are exposed per memory bank, and expand the MCP tool set with additional tools and parameters. ([`3ffec650`](https://github.com/vectorize-io/hindsight/commit/3ffec650))
- Enable the bank configuration API by default. ([`4d030707`](https://github.com/vectorize-io/hindsight/commit/4d030707))
- Support filtering graph-based memory retrieval by tags. ([`0bb5ca4c`](https://github.com/vectorize-io/hindsight/commit/0bb5ca4c))
- Add batch observations consolidation to process multiple observations more efficiently. ([`0aa7c2b3`](https://github.com/vectorize-io/hindsight/commit/0aa7c2b3))
- Add OpenClaw options to toggle autoRecall and exclude specific providers. ([`3f9eb27c`](https://github.com/vectorize-io/hindsight/commit/3f9eb27c))
- Add a ZeroEntropy reranker provider option. ([`17259675`](https://github.com/vectorize-io/hindsight/commit/17259675))

**Improvements**

- Increase customization options for reflect, retain, and consolidation behavior. ([`2a322732`](https://github.com/vectorize-io/hindsight/commit/2a322732))
- Include source document metadata in fact extraction results. ([`87219b73`](https://github.com/vectorize-io/hindsight/commit/87219b73))

**Bug Fixes**

- Raise a clear error when embedding dimensions exceed pgvector HNSW limits (instead of failing later at runtime). ([`8cd65b98`](https://github.com/vectorize-io/hindsight/commit/8cd65b98))
- Fix multi-tenant schema isolation issues in storage and the bank config API. ([`b180b3ad`](https://github.com/vectorize-io/hindsight/commit/b180b3ad))
- Ensure LiteLLM embedding calls use the correct float encoding format to prevent embedding failures. ([`58f2de70`](https://github.com/vectorize-io/hindsight/commit/58f2de70))
- Improve recall performance by reducing memory usage during retrieval. ([`9f0c031d`](https://github.com/vectorize-io/hindsight/commit/9f0c031d))
- Handle observation regeneration correctly when underlying memories are deleted. ([`ac9a94ad`](https://github.com/vectorize-io/hindsight/commit/ac9a94ad))
- Fix reflect retrieval to correctly populate dependencies and enforce full hierarchical retrieval. ([`8b1a4658`](https://github.com/vectorize-io/hindsight/commit/8b1a4658))
- Fix OpenClaw health checks by passing the auth token to the health endpoint. ([`40b02645`](https://github.com/vectorize-io/hindsight/commit/40b02645))

## [0.4.13](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.13)

**Features**

- Switched the default OpenAI LLM to gpt-4o-mini. ([`325b5cc1`](https://github.com/vectorize-io/hindsight/commit/325b5cc1))
- Observation recall now includes the source facts behind recalled observations. ([`5569d4ad`](https://github.com/vectorize-io/hindsight/commit/5569d4ad))
- Added CrewAI integration to enable persistent memory. ([`41db2960`](https://github.com/vectorize-io/hindsight/commit/41db2960))

**Bug Fixes**

- Fixed npx hindsight-control-plane failing to run. ([`0758827d`](https://github.com/vectorize-io/hindsight/commit/0758827d))
- Improved MCP compatibility by aligning the local MCP implementation with the server and removing the deprecated stateless parameter. ([`ea8163c5`](https://github.com/vectorize-io/hindsight/commit/ea8163c5))
- Fixed Docker startup failures when using named Docker volumes. ([`ac739487`](https://github.com/vectorize-io/hindsight/commit/ac739487))
- Prevented reranker crashes when an upstream provider returns an error. ([`58c4d657`](https://github.com/vectorize-io/hindsight/commit/58c4d657))
- Improved accuracy of fact temporal ordering by reducing per-fact time offsets. ([`c3ef1555`](https://github.com/vectorize-io/hindsight/commit/c3ef1555))
- Client timeout settings are now properly respected. ([`dcaa9f14`](https://github.com/vectorize-io/hindsight/commit/dcaa9f14))
- Fixed documents not being tracked when fact extraction returns zero facts. ([`f78278ea`](https://github.com/vectorize-io/hindsight/commit/f78278ea))

## [0.4.12](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.12)

**Features**

- Accept and ingest PDFs, images, and common Office documents as inputs. ([`224b7b74`](https://github.com/vectorize-io/hindsight/commit/224b7b74))
- Add the Iris file parser for improved document parsing support. ([`7eafba66`](https://github.com/vectorize-io/hindsight/commit/7eafba66))
- Add async Retain support via provider Batch APIs (e.g., OpenAI and Groq) for higher-throughput ingestion. ([`40d42c58`](https://github.com/vectorize-io/hindsight/commit/40d42c58))
- Allow Recall to return chunks only (no memories) by setting max_tokens=0. ([`7dad9da0`](https://github.com/vectorize-io/hindsight/commit/7dad9da0))
- Add a Go client SDK for the Hindsight API. ([`2a47389f`](https://github.com/vectorize-io/hindsight/commit/2a47389f))
- Add support for the pgvectorscale (DiskANN) vector index backend. ([`95c42204`](https://github.com/vectorize-io/hindsight/commit/95c42204))
- Add support for Azure pg_diskann vector indexing. ([`476726c2`](https://github.com/vectorize-io/hindsight/commit/476726c2))

**Improvements**

- Improve reliability of async batch Retain when ingesting large payloads. ([`aefb3fcf`](https://github.com/vectorize-io/hindsight/commit/aefb3fcf))
- Improve AI SDK tooling to make it easier to work with Hindsight programmatically. ([`d06a0259`](https://github.com/vectorize-io/hindsight/commit/d06a0259))

**Bug Fixes**

- Ensure document tags are preserved when using the async Retain flow. ([`b4b5c44a`](https://github.com/vectorize-io/hindsight/commit/b4b5c44a))
- Fix OpenClaw ingestion failures for very large content (E2BIG). ([`6bad6673`](https://github.com/vectorize-io/hindsight/commit/6bad6673))
- Harden OpenClaw behavior (safer shell usage, better HTTP mode handling, and more reliable initialization), including per-user banks support. ([`c4610130`](https://github.com/vectorize-io/hindsight/commit/c4610130))
- Improve Python client async API consistency and reduce connection drop issues via keepalive timeout fixes. ([`8114ef44`](https://github.com/vectorize-io/hindsight/commit/8114ef44))

## [0.4.11](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.11)

**Features**

- Added support for LiteLLM SDK as an embeddings and reranking provider. ([`e408b7e`](https://github.com/vectorize-io/hindsight/commit/e408b7e))
- Expanded Postgres search support with additional text/vector extensions, including TimescaleDB pg_textsearch and vchord/pgvector options. ([`d871c30`](https://github.com/vectorize-io/hindsight/commit/d871c30))
- Added hierarchical configuration scopes (system, tenant, bank) for more flexible multi-tenant setup and overrides. ([`8d731f2`](https://github.com/vectorize-io/hindsight/commit/8d731f2))
- Added reverse proxy/base-path support for running Hindsight behind a proxy. ([`93ddd41`](https://github.com/vectorize-io/hindsight/commit/93ddd41))
- Added MCP tools to create, read, update, and delete mental models. ([`f641b30`](https://github.com/vectorize-io/hindsight/commit/f641b30))
- Added a "docs" skill for agents/tools to access documentation-oriented capabilities. ([`dd1e098`](https://github.com/vectorize-io/hindsight/commit/dd1e098))
- Added an OpenClaw configuration option to skip recall/retain for specific providers. ([`fb7be3e`](https://github.com/vectorize-io/hindsight/commit/fb7be3e))

**Improvements**

- Improved LiteLLM gateway model configuration for more reliable provider/model selection. ([`7d95a00`](https://github.com/vectorize-io/hindsight/commit/7d95a00))
- Exposed actual LLM token usage in retain results to improve cost/usage visibility. ([`83ca669`](https://github.com/vectorize-io/hindsight/commit/83ca669))
- Added user-initiated attribution to request context to improve async task and usage attribution. ([`90be7c6`](https://github.com/vectorize-io/hindsight/commit/90be7c6))
- Added OpenTelemetry tracing for improved request traceability and observability. ([`69dec8e`](https://github.com/vectorize-io/hindsight/commit/69dec8e))
- Helm chart: split TEI embedding and reranker into separate deployments for independent scaling and rollout. ([`43f9a8b`](https://github.com/vectorize-io/hindsight/commit/43f9a8b))
- Helm chart: added PodDisruptionBudgets and per-component affinity controls for more resilient scheduling. ([`9943957`](https://github.com/vectorize-io/hindsight/commit/9943957))

**Bug Fixes**

- Fixed a recursion issue in memory retention that could cause failures or runaway memory usage. ([`4f11210`](https://github.com/vectorize-io/hindsight/commit/4f11210))
- Fixed Reflect API serialization/schema issues for "based_on" so reflections are returned and stored correctly. ([`f9a8a8e`](https://github.com/vectorize-io/hindsight/commit/f9a8a8e))
- Improved MCP server compatibility by allowing extra tool arguments when appropriate and fixing bank ID resolution priority. ([`7ee229b`](https://github.com/vectorize-io/hindsight/commit/7ee229b))
- Added missing trust_code environment configuration support. ([`60574ee`](https://github.com/vectorize-io/hindsight/commit/60574ee))
- Hardened the MCP server with fixes to routing/validation and more accurate usage metering. ([`e798979`](https://github.com/vectorize-io/hindsight/commit/e798979))
- Fixed the slim Docker image to include tiktoken to prevent runtime tokenization errors. ([`6eec83b`](https://github.com/vectorize-io/hindsight/commit/6eec83b))
- Fixed MCP operations not being tracked correctly for usage metering. ([`888b50d`](https://github.com/vectorize-io/hindsight/commit/888b50d))
- Helm chart: fixed GKE deployments overriding the configured HINDSIGHT_API_PORT. ([`03f47e2`](https://github.com/vectorize-io/hindsight/commit/03f47e2))

## [0.4.10](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.10)

**Features**

- Provided a slimmer Docker distribution to reduce image size and speed up pulls. ([`f648178`](https://github.com/vectorize-io/hindsight/commit/f648178))
- Added Markdown support in Reflect and Mental Models content. ([`c4ef090`](https://github.com/vectorize-io/hindsight/commit/c4ef090))
- Added built-in Supabase tenant extension for running Hindsight with Supabase-backed multi-tenancy. ([`e99ee0f`](https://github.com/vectorize-io/hindsight/commit/e99ee0f))
- Added TenantExtension authentication support to the MCP endpoint. ([`fedfb49`](https://github.com/vectorize-io/hindsight/commit/fedfb49))

**Improvements**

- Improved MCP tool availability/routing based on the endpoint being used. ([`d90588b`](https://github.com/vectorize-io/hindsight/commit/d90588b))

**Bug Fixes**

- Stopped logging database usernames and passwords to prevent credential leaks in logs. ([`c568094`](https://github.com/vectorize-io/hindsight/commit/c568094))
- Fixed OpenClaw sessions wiping memory on each new session. ([`981cf60`](https://github.com/vectorize-io/hindsight/commit/981cf60))
- Fixed hindsight-embed profiles not loading correctly. ([`0430588`](https://github.com/vectorize-io/hindsight/commit/0430588))
- Fixed tagged directives so they correctly apply to tagged mental models. ([`278718d`](https://github.com/vectorize-io/hindsight/commit/278718d))
- Fixed a cast error that could cause failures at runtime. ([`093ecff`](https://github.com/vectorize-io/hindsight/commit/093ecff))

**Other**

- Added a docker-compose example to simplify local deployment and testing. ([`5179d5f`](https://github.com/vectorize-io/hindsight/commit/5179d5f))

## [0.4.9](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.9)

**Features**

- New AI SDK integration. ([`7e339e1`](https://github.com/vectorize-io/hindsight/commit/7e339e1))
- Add a Python SDK for running Hindsight in embedded mode (HindsightEmbedded). ([`d3302c9`](https://github.com/vectorize-io/hindsight/commit/d3302c9))
- Add streaming support to the hindsight-litellm wrappers. ([`665877b`](https://github.com/vectorize-io/hindsight/commit/665877b))
- Add OpenClaw support for connecting to an external Hindsight API and using dynamic per-channel memory banks. ([`6b34692`](https://github.com/vectorize-io/hindsight/commit/6b34692))

**Improvements**

- Improve the mental models experience in the control plane UI. ([`7097716`](https://github.com/vectorize-io/hindsight/commit/7097716))
- Reduce noisy Hugging Face logging output. ([`34d9188`](https://github.com/vectorize-io/hindsight/commit/34d9188))

**Bug Fixes**

- Improve recall endpoint reliability by handling timeouts correctly and rejecting overly long queries. ([`dd621a6`](https://github.com/vectorize-io/hindsight/commit/dd621a6))
- Improve /reflect behavior with Claude Code and Codex providers. ([`a43d208`](https://github.com/vectorize-io/hindsight/commit/a43d208))
- Fix OpenClaw shell argument escaping for more reliable command execution. ([`63e2964`](https://github.com/vectorize-io/hindsight/commit/63e2964))

## [0.4.8](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.8)

**Features**

- Added profile support for `hindsight-embed`, enabling separate embedding configurations/workspaces. ([`6c7f057`](https://github.com/vectorize-io/hindsight/commit/6c7f057))
- Added support for additional LLM backends, including OpenAI Codex and Claude Code. ([`539190b`](https://github.com/vectorize-io/hindsight/commit/539190b))

**Improvements**

- Enhanced OpenClaw and `hindsight-embed` parameter/config options for easier configuration and better defaults. ([`749478d`](https://github.com/vectorize-io/hindsight/commit/749478d))
- Added OpenClaw plugin configuration options to select LLM provider and model. ([`8564135`](https://github.com/vectorize-io/hindsight/commit/8564135))
- Server now prints its version during startup to simplify debugging and support requests. ([`1499ce5`](https://github.com/vectorize-io/hindsight/commit/1499ce5))
- Improved tracing/debuggability by propagating request context through asynchronous background tasks. ([`44d9125`](https://github.com/vectorize-io/hindsight/commit/44d9125))
- Added stronger validation and context for mental model create/refresh operations to prevent invalid requests. ([`35127d5`](https://github.com/vectorize-io/hindsight/commit/35127d5))

**Bug Fixes**

- Improved embedding CLI experience with richer logs and isolated profiles to avoid cross-contamination between runs. ([`794a743`](https://github.com/vectorize-io/hindsight/commit/794a743))
- Operation validation now runs correctly in the worker process, preventing invalid background operations from slipping through. ([`96f0e54`](https://github.com/vectorize-io/hindsight/commit/96f0e54))
- Fixed unreliable behavior when using a custom PostgreSQL schema. ([`3825506`](https://github.com/vectorize-io/hindsight/commit/3825506))

## [0.4.7](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.7)

**Features**

- Add extension hooks to validate and customize mental model operations. ([`9c3fda7`](https://github.com/vectorize-io/hindsight/commit/9c3fda7))
- Add support for using an external embedding API provider in OpenClaw plugin (with additional OpenClaw compatibility fixes). ([`4b57b82`](https://github.com/vectorize-io/hindsight/commit/4b57b82))

**Improvements**

- Speed up container startup by preloading the tiktoken encoding during Docker image builds. ([`039944c`](https://github.com/vectorize-io/hindsight/commit/039944c))

**Bug Fixes**

- Prevent PostgreSQL insert failures by stripping null bytes from text fields before saving. ([`ef9d3a1`](https://github.com/vectorize-io/hindsight/commit/ef9d3a1))
- Fix worker schema selection so it uses the correct default database schema. ([`d788a55`](https://github.com/vectorize-io/hindsight/commit/d788a55))
- Honor an already-set HINDSIGHT_API_DATABASE_URL instead of overwriting it in the hindsight-embed workflow. ([`f0cb192`](https://github.com/vectorize-io/hindsight/commit/f0cb192))

## [0.4.6](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.6)

**Improvements**

- Improved OpenClaw configuration setup to make embedding integration easier to configure. ([`27498f9`](https://github.com/vectorize-io/hindsight/commit/27498f9))

**Bug Fixes**

- Fixed OpenClaw embedding version binding/versioning to prevent mismatches when using the embed integration. ([`1163b1f`](https://github.com/vectorize-io/hindsight/commit/1163b1f))

## [0.4.5](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.5)

**Bug Fixes**

- Fixed occasional failures when retaining memories asynchronously with timestamps. ([`cbb8fc6`](https://github.com/vectorize-io/hindsight/commit/cbb8fc6))

## [0.4.4](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.4)

**Bug Fixes**

- Fixed async “retain” operations failing when a timestamp is provided. ([`35f0984`](https://github.com/vectorize-io/hindsight/commit/35f0984))
- Corrected the OpenClaw daemon integration name to “openclaw” (previously “openclawd”). ([`b364bc3`](https://github.com/vectorize-io/hindsight/commit/b364bc3))

## [0.4.3](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.3)

**Features**

- Add Vertex AI as a supported LLM provider. ([`c2ac7d0`](https://github.com/vectorize-io/hindsight/commit/c2ac7d0), [`49ae55a`](https://github.com/vectorize-io/hindsight/commit/49ae55a))
- Add Bearer token authentication for MCP and propagate tenant authentication across MCP requests. ([`0da77ce`](https://github.com/vectorize-io/hindsight/commit/0da77ce))

**Improvements**

- CLI: add a --wait flag for consolidate and a --date filter for listing documents. ([`ff20bf9`](https://github.com/vectorize-io/hindsight/commit/ff20bf9))

**Bug Fixes**

- Fix worker polling deadlocks to prevent background processing from stalling. ([`f4f86e3`](https://github.com/vectorize-io/hindsight/commit/f4f86e3))
- Improve reliability of Docker builds by retrying ML model downloads. ([`ecc590c`](https://github.com/vectorize-io/hindsight/commit/ecc590c))
- Fix tenant authentication handling for internal background tasks and ensure the control-plane forwards required auth to the dataplane. ([`03bf13e`](https://github.com/vectorize-io/hindsight/commit/03bf13e))
- Ensure tenant database migrations run at startup and workers use the correct tenant schema context. ([`657fe02`](https://github.com/vectorize-io/hindsight/commit/657fe02))
- Fix control-plane graph endpoint errors when upstream data is missing. ([`751f99a`](https://github.com/vectorize-io/hindsight/commit/751f99a))

**Other**

- Rename the default bot/user identity from "moltbot" to "openclaw". ([`728ce13`](https://github.com/vectorize-io/hindsight/commit/728ce13))

## [0.4.2](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.2)

**Features**

- Added Clawdbot/Moltbot/OpenClaw integration. ([`12e9a3d`](https://github.com/vectorize-io/hindsight/commit/12e9a3d))

**Improvements**

- Added additional configuration options to control LLM retry behavior. ([`3f211f0`](https://github.com/vectorize-io/hindsight/commit/3f211f0))
- Added real-time logs showing a detailed timing breakdown during consolidation runs. ([`8781c9f`](https://github.com/vectorize-io/hindsight/commit/8781c9f))

**Bug Fixes**

- Fixed hindsight-embed crashing on macOS. ([`c16ccc2`](https://github.com/vectorize-io/hindsight/commit/c16ccc2))

## [0.4.1](https://github.com/vectorize-io/hindsight/releases/tag/v0.4.1)

**Features**

- Added support for using a non-default PostgreSQL schema by default. ([`2b72e1f`](https://github.com/vectorize-io/hindsight/commit/2b72e1f))

**Improvements**

- Improved memory consolidation performance (benchmarking and optimizations). ([`b43ef98`](https://github.com/vectorize-io/hindsight/commit/b43ef98))

**Bug Fixes**

- Fixed the /version endpoint returning an incorrect version. ([`cfcc23c`](https://github.com/vectorize-io/hindsight/commit/cfcc23c))
- Fixed mental model search failing due to UUID type mismatch after text-ID migration. ([`94cc0a1`](https://github.com/vectorize-io/hindsight/commit/94cc0a1))
- Added safer PyTorch device detection to prevent crashes on some environments. ([`67c4788`](https://github.com/vectorize-io/hindsight/commit/67c4788))
- Fixed Python packages exposing an incorrect __version__ value. ([`fccbdfe`](https://github.com/vectorize-io/hindsight/commit/fccbdfe))

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
