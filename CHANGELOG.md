# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- OpenClaw plugin: Dynamic memory banks for per-channel isolation
  - Each channel gets its own bank: `{messageProvider}-{channelId}` (e.g., `slack-C123`, `telegram-456`)
  - `dynamicBankId` config option (default: true)
  - `bankIdPrefix` config option for namespace prefixing
  - Bank mission auto-set on first use of each bank
  - Hook handlers updated to use correct `(event, ctx)` signature
- OpenClaw plugin: External Hindsight API support
  - `HINDSIGHT_EMBED_API_URL` / `hindsightApiUrl` config to connect to remote Hindsight
  - `HINDSIGHT_EMBED_API_TOKEN` / `hindsightApiToken` for authentication
  - Skip local daemon when external API configured

### Fixed
- OpenClaw plugin: Shell argument escaping for special characters (`?`, `!`, `$`, backticks, etc.)
  - Added comprehensive `escapeShellArg()` function using POSIX single-quote escaping
  - 17 new tests covering all shell-special character scenarios

### Added
- Synced with upstream v0.4.7
  - Mental model extension hooks (pre/post validation for billing/quota)
  - hindsight-embed external API support (connect to remote servers)
  - OpenClaw fixes (port, daemon recovery, OpenRouter, macOS crashes)
  - Our PRs merged: null bytes sanitization (#238), tiktoken preload (#249)
  - Worker default schema fix for multi-tenant setups
- Synced with upstream v0.4.6
  - Bug fixes: deadlock in worker polling, retain async timestamp issues
  - OpenClaw: moltbot → openclaw rename with config setup
  - Our PRs merged back: MCP auth, docker retry, CLI flags, control-plane fixes
  - Docs: VertexAI, MCP, OpenClaw, embed page

### Fixed
- CI: Build job no longer requires production approval (only deploy does)
- Docker: Added retry logic for ML model downloads (3 retries with exponential backoff)
- Docker: Preload tiktoken encoding during build (avoids runtime download)

### Added
- Synced with upstream v0.4.2 (12 commits merged)
  - Vertex AI as LLM provider
  - Per-operation LLM retry/backoff configuration
  - Moltbot (Claude Code) integration
  - Consolidation performance benchmark and timing breakdown
  - Worker poller hardening for tenant schemas
  - hindsight-embed macOS crash fix
  - /version endpoint fix
- CLI: `--wait` flag for `bank consolidate` to poll for completion
- CLI: `--poll-interval` option for consolidate (default 10s)
- CLI: `--date` filter for `document list` (yesterday/today/YYYY-MM-DD/all)

### Changed
- Merged 4 hindsight skills into 1 unified skill (515 → 249 lines)
  - Combined dev guidance + memory usage (local/cloud/self-hosted)
  - New `references/memory-usage.md` for memory patterns
  - Deleted `hindsight-local/`, `hindsight-cloud/`, `hindsight-self-hosted/`

### Added
- Dev environment GHA workflow (`deploy-gcp-dev.yml`) for `develop` branch
  - Deploys to `hindsight-dev` namespace with `beta`/`beta-{sha}` image tags
  - Auto-deploys without manual approval (unlike production)
- Auto-deploy to GKE in GitHub Actions workflow after Docker image push
  - Authenticates to GCP, gets GKE credentials, restarts deployment with `kubectl rollout restart`
  - Targets `gcp-k8s-xsolla-n8n-prod` cluster / `hindsight` namespace / `hindsight-current` deployment
  - Waits up to 5 min for rollout completion
- Docker image published to GCP Artifact Registry (`us-docker.pkg.dev/xsolla-n8n-prod/hindsight/hindsight`)
- `api_key_resolver` field on `MCPToolsConfig` for tenant auth token propagation
- `async_processing` parameter to MCP `retain` tool for non-blocking memory storage
  - Default: `True` (queues for background processing)
  - Set to `False` to wait for completion
- `list_memories` MCP tool for full-text search with pagination
  - Parameters: `type`, `q`, `limit`, `offset`, `bank_id`
  - Complements semantic `recall` tool with exact-match search

### Changed
- Improved CLAUDE.md documentation with memory types and architecture details

### Fixed
- Background worker auth failure for async retain on public schema
  - `_authenticate_tenant` checked `current != "public"` for internal requests, causing public-schema background tasks to fail with "Invalid API key"
  - Removed the public-schema guard — internal tasks are already authenticated at submission time
- Control plane GUI completely broken when tenant extension is active
  - Proxy routes never sent Authorization header to dataplane API
  - Added `HINDSIGHT_CP_DATAPLANE_API_KEY` env var for control plane → dataplane auth
  - Updated all SDK clients and 7 direct-fetch route files with auth headers
- Control plane graph route crash with "Value is not JSON serializable"
  - SDK returned `undefined` for `response.data` on error responses
  - Added error check before `NextResponse.json()` serialization
- Docker container crash caused by tiktoken runtime download failure through VPN
  - Pre-cache tiktoken cl100k_base encoding in Docker image build
  - Eliminates runtime network dependency for container startup
- DNS resolution issues through gluetun VPN tunnel
  - Switch from Cloudflare (1.1.1.1) to Google DNS (8.8.8.8)
  - Add 60-second startup delay for network service readiness
  - Lower WireGuard MTU to 1280 for stability
- Gluetun health check failures caused by VPN DNS timeouts
  - Use IP-based health targets (`HEALTH_TARGET_ADDRESSES=1.1.1.1:443,8.8.8.8:443`)
  - Eliminates DNS dependency for health checks
- MCP tools failing with "Invalid API key" when tenant extension enabled
  - Bearer token was validated in middleware but not propagated to tools
  - Added `_current_api_key` context variable to pass token through to `RequestContext`
- API startup hang caused by reranker model mismatch
  - Dockerfile caches `ms-marco-MiniLM-L-6-v2`, .env had `L-12-v2`
  - API hung trying to download uncached model through VPN
  - Aligned .env to use cached L-6 model

## [0.2.1] - 2026-01-06

### Added
- Multi-bank support for MCP tools via `bank_id` parameter
- `reflect` MCP tool for LLM-based reasoning over memories
- `list_banks` MCP tool to discover available memory banks
- `create_bank` MCP tool to create new memory banks
- `X-Bank-Id` header support for Claude Code compatibility

### Fixed
- MCP HTTP endpoint 404 error (corrected `http_app` path parameter)

## [0.2.0] - 2025-12-XX

### Added
- Initial MCP server implementation with FastMCP
- `retain` and `recall` MCP tools
- MCPMiddleware for bank_id extraction from path or header

---

_For older versions, see [GitHub Releases](https://github.com/vectorize-io/hindsight/releases)_
