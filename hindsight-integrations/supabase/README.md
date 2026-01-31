# Supabase Tenant Extension for Hindsight

A built-in TenantExtension that validates [Supabase](https://supabase.com) JWTs and provides multi-tenant memory isolation. Each authenticated user gets their own PostgreSQL schema, ensuring complete data separation.

## Features

- **Local JWT Verification** - Validates tokens locally using JWKS public keys (no network call per request)
- **Automatic Schema Isolation** - Each user gets `{prefix}_{user_id}` schema
- **Zero User Management** - Leverages your existing Supabase Auth setup
- **Production Ready** - Includes health checks, timeouts, key rotation handling, and error handling
- **Built-in** - Ships with Hindsight, no extra installation needed
- **Legacy Support** - Falls back to `/auth/v1/user` endpoint for HS256 projects

## Configuration

The Supabase tenant extension is built into Hindsight. Just set the environment variables:

```bash
# Required
HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.supabase_tenant:SupabaseTenantExtension
HINDSIGHT_API_TENANT_SUPABASE_URL=https://your-project.supabase.co

# Optional - only needed for legacy HS256 projects or startup health check
HINDSIGHT_API_TENANT_SUPABASE_SERVICE_KEY=your-service-role-key

# Optional
HINDSIGHT_API_TENANT_SCHEMA_PREFIX=user  # Default: "user"
```

> **Note:** Most Supabase projects use asymmetric JWT signing (ES256/RS256) and the extension verifies tokens locally using JWKS — no service key needed. The `service_role` key is only required if your project uses legacy HS256 signing or if you want the startup health check.

## Usage

Clients pass their Supabase access token in the Authorization header:

```bash
# Get user's access token from Supabase Auth
TOKEN=$(curl -s -X POST "https://your-project.supabase.co/auth/v1/token?grant_type=password" \
  -H "apikey: your-anon-key" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "xxx"}' | jq -r '.access_token')

# Use with Hindsight
curl -X POST "https://your-hindsight-server/v1/default/banks/my-bank/memories" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"items": [{"content": "User preference: likes dark mode"}]}'
```

## How It Works

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Your App      │     │   Hindsight     │     │   Supabase      │
│                 │     │                 │     │                 │
│  1. User logs   │     │                 │     │  JWKS keys      │
│     in via      │────▶│                 │     │  fetched once   │
│     Supabase    │     │                 │     │  on startup     │
│                 │     │                 │     │                 │
│  2. App calls   │     │  3. Extension   │     │                 │
│     Hindsight   │────▶│     verifies    │     │                 │
│     with JWT    │     │     JWT locally │     │                 │
│                 │     │     (no network │     │                 │
│                 │     │      call)      │     │                 │
│                 │     │                 │     │                 │
│                 │     │  4. Routes to   │     │                 │
│                 │◀────│     user's      │     │                 │
│                 │     │     schema      │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

1. On startup, Hindsight fetches JWKS public keys from Supabase (cached for 10 minutes)
2. User authenticates with your app via Supabase Auth
3. Your app calls Hindsight API with the user's JWT
4. Extension verifies the JWT signature locally using cached public keys
5. On success, routes request to user's isolated schema (`user_{uuid}`)

For legacy HS256 projects, the extension falls back to calling `/auth/v1/user` per request.

## Schema Isolation

Each user gets a completely isolated PostgreSQL schema:

```
Hindsight Database
├── Schema: user_abc123_def456 (User A)
│   ├── memories
│   ├── entities
│   └── ...
├── Schema: user_xyz789_... (User B)
│   ├── memories
│   ├── entities
│   └── ...
└── Schema: public (Hindsight internals)
```

User A cannot access User B's data - they're in separate schemas.

## Configuration Options

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HINDSIGHT_API_TENANT_SUPABASE_URL` | Yes | - | Your Supabase project URL |
| `HINDSIGHT_API_TENANT_SUPABASE_SERVICE_KEY` | No | - | Supabase service_role key (only needed for HS256 projects or health check) |
| `HINDSIGHT_API_TENANT_SCHEMA_PREFIX` | No | `user` | Prefix for schema names (must be a valid Postgres identifier) |

## Deployment Examples

### Docker

```dockerfile
FROM ghcr.io/vectorize-io/hindsight:latest

ENV HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.supabase_tenant:SupabaseTenantExtension
```

### Railway

```toml
# railway.toml
[build]
builder = "dockerfile"
dockerfilePath = "Dockerfile"

[deploy]
healthcheckPath = "/health"
```

Set environment variables in Railway dashboard.

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `401 Unauthorized` | Invalid or expired JWT | Get fresh token from Supabase |
| `Missing Authorization header` | No Bearer token sent | Add `Authorization: Bearer <token>` header |
| `Unable to find signing key` | JWT signed with unknown key | Check Supabase JWT algorithm settings |
| `Authentication timeout` | Supabase slow/unreachable (legacy mode) | Check Supabase status, retry |
| `SUPABASE_SERVICE_KEY is required when JWKS is not available` | HS256 project without service key | Provide service_role key or switch to asymmetric JWT signing |

## Contributing

This extension was originally developed by [BrighterBalance](https://brighterbalance.app) for their AI advisor product.

## License

MIT
