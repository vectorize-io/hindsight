# Cloudflare OAuth Proxy for Self-Hosted Hindsight

An [OAuth 2.1](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1-12) proxy that connects cloud-based MCP clients (such as [claude.ai](https://claude.ai), [Claude Code](../claude-code/), and [Codex](../codex/)) to a self-hosted [Hindsight](https://vectorize.io/hindsight) instance. Built on [Cloudflare Workers](https://developers.cloudflare.com/workers/) using the [`@cloudflare/workers-oauth-provider`](https://github.com/cloudflare/workers-oauth-provider) library.

## Why

Cloud-based MCP clients require an OAuth 2.1 flow to connect to remote MCP servers. Self-hosted Hindsight instances typically sit behind a private network or Cloudflare Tunnel, and don't natively expose an OAuth endpoint. This Worker bridges that gap: it handles the OAuth dance on a public domain, authenticates the user with a simple password gate, and proxies authenticated MCP traffic to your Hindsight origin through a Cloudflare Tunnel.

## Architecture

```
Cloud MCP Client (claude.ai, Claude Code, Codex)
        |
        | HTTPS + OAuth 2.1
        v
Cloudflare Worker (this proxy)
   - OAuth 2.1 authorization server
   - Dynamic client registration (RFC 7591)
   - PKCE (S256 only)
   - CORS restricted to allowlisted origins
        |
        | HTTPS + Cloudflare Tunnel
        v
Self-hosted Hindsight (Docker)
```

## Prerequisites

- A [Cloudflare](https://www.cloudflare.com/) account with a domain
- A running self-hosted Hindsight instance (see [self-hosting quickstart](https://vectorize.io/hindsight/quickstart/self-hosting))
- A [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) exposing your Hindsight instance
- [Node.js](https://nodejs.org/) (v18+)

## Setup

### 1. Install dependencies

```bash
cd hindsight-integrations/cloudflare-oauth-proxy
npm install
```

### 2. Configure `wrangler.toml`

Copy the included `wrangler.toml` and update the placeholder values:

- `HINDSIGHT_ORIGIN`: Your Cloudflare Tunnel origin URL (e.g., `https://hindsight-origin.yourdomain.com`)
- `kv_namespaces.id`: Create a KV namespace via `npx wrangler kv namespace create OAUTH_KV` and paste the returned ID
- `routes.pattern` / `routes.zone_name`: Your public-facing domain for the proxy

### 3. Set secrets

```bash
npx wrangler secret put SESSION_SECRET       # Password for the login page
npx wrangler secret put PROXY_SECRET         # X-Proxy-Secret header value (must match your origin WAF rule)
npx wrangler secret put HINDSIGHT_API_TOKEN  # Bearer token for the Hindsight API
npx wrangler secret put ALLOWED_EMAIL        # Your email (used as the OAuth user identity)
```

### 4. Deploy

```bash
npm run deploy
```

### 5. Secure your origin

Add a WAF rule on your Cloudflare Tunnel origin hostname to block requests that don't carry the correct `X-Proxy-Secret` header. This ensures only the Worker can reach your Hindsight instance.

## Connecting clients

Once deployed, add the Worker URL as a remote MCP server in your client:

- **claude.ai**: Settings > MCP Servers > Add > enter `https://hindsight.yourdomain.com/mcp`
- **Claude Code**: `claude mcp add hindsight-remote https://hindsight.yourdomain.com/mcp --transport http`
- **Codex**: Configure via the MCP settings with the same URL

On first connection, you'll be redirected to a login page. Enter the `SESSION_SECRET` password to authorize. This only needs to happen once per OAuth session.

## Secrets reference

| Secret | Purpose |
|--------|---------|
| `SESSION_SECRET` | Password shown on the login page to authorize a session |
| `PROXY_SECRET` | Value sent as `X-Proxy-Secret` header to the origin (for WAF validation) |
| `HINDSIGHT_API_TOKEN` | Bearer token for authenticating with the Hindsight API |
| `ALLOWED_EMAIL` | Your email address, used as the OAuth user identity |

## Security notes

- **Single-user design.** Anyone who knows `SESSION_SECRET` will be authorized as `ALLOWED_EMAIL`. Use a high-entropy secret and keep it private.
- CORS is restricted to `claude.ai` origins by default. To support additional clients (e.g., ChatGPT), add their origins to the `ALLOWED_ORIGINS` set in `src/cors.ts`.
- The OAuth authorization-server metadata is rewritten to advertise `code_challenge_methods_supported = ["S256"]`. Actual enforcement of S256 at the token endpoint is delegated to `@cloudflare/workers-oauth-provider`.
- The login-page password check uses a constant-time (SHA-256 based) comparison to resist timing attacks.
- OAuth state is stored in Cloudflare KV with a 5-minute TTL and is deleted on consumption.
- The proxy strips the client's `Authorization` header (and any attempted `X-Proxy-Secret`) and replaces them with the server's configured values before forwarding to the origin.
- Upstream response headers are filtered to a whitelist (`content-type`, `cache-control`, `mcp-session-id`, etc.) so the origin can't leak cookies or its own CORS headers through the proxy.

## Development

```bash
npm install
npm run typecheck
npm test
```

Tests use [vitest](https://vitest.dev/) and run in a plain Node environment — the Workers-specific modules are isolated in `src/index.ts` so the testable units can be imported without the Cloudflare Workers runtime.
