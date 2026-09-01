# hindsight-ext-supabase-tenant

A Hindsight `TenantExtension` that authenticates requests with [Supabase](https://supabase.com)
Auth JWTs and gives every user their own PostgreSQL schema, so memories are isolated
at the database level.

- **Local JWT verification** using the project's JWKS public keys — no network call per
  request. Falls back to `/auth/v1/user` for legacy HS256 projects.
- **Schema per user**: a user with id `a1b2…7890` gets the schema `user_a1b2…7890`
  (hyphens become underscores), migrated on first access and cached afterwards.
- **No user management**: your existing Supabase project is the source of identity.

> This extension shipped inside the Hindsight server up to **0.9.2** as
> `hindsight_api.extensions.builtin.supabase_tenant`. It is now a separate package —
> see [Migrating](#migrating-from-the-built-in-extension).

## Install

Install it into the same environment as the Hindsight server:

```bash
pip install hindsight-ext-supabase-tenant
```

Requires `hindsight-api-slim >= 0.9.2`. For Docker, see the [Dockerfile](./Dockerfile)
and the [packaging guide](../README.md#docker-packaging).

## Configure

```bash
HINDSIGHT_API_TENANT_EXTENSION=hindsight_ext_supabase_tenant:SupabaseTenantExtension
HINDSIGHT_API_TENANT_SUPABASE_URL=https://xxx.supabase.co
```

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `HINDSIGHT_API_TENANT_SUPABASE_URL` | yes | — | Supabase project URL |
| `HINDSIGHT_API_TENANT_SUPABASE_SERVICE_KEY` | only for HS256 projects | — | `service_role` key. Needed when JWKS is unavailable, and used for the startup health check |
| `HINDSIGHT_API_TENANT_SCHEMA_PREFIX` | no | `user` | Schema name prefix; must be a valid Postgres identifier |

If your project uses legacy HS256 signing and no service key is set, the server fails
at startup rather than accepting unverifiable tokens.

Give the API and the worker the **same** variables: the worker calls `list_tenants()`
to decide which schemas to consolidate, so a worker without the extension leaves every
tenant's background processing stopped.

## Use

Clients send their Supabase JWT as a bearer token:

```bash
curl -H "Authorization: Bearer <supabase_jwt>" \
  https://your-hindsight-server/v1/default/banks/my-bank/memories/recall
```

## Migrating from the built-in extension

Installs that set the old path get a startup error naming these two steps:

```diff
-HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.supabase_tenant:SupabaseTenantExtension
+HINDSIGHT_API_TENANT_EXTENSION=hindsight_ext_supabase_tenant:SupabaseTenantExtension
```

plus `pip install hindsight-ext-supabase-tenant`. Every `HINDSIGHT_API_TENANT_*`
setting keeps its name and meaning, the schema naming is unchanged, and existing
tenant schemas are picked up as they were before — this is a packaging move, not a
behaviour change.

## Develop

```bash
uv sync --extra dev
uv run pytest tests -v
```

## License

MIT. Originally contributed by [BrighterBalance](https://brighterbalance.app).
