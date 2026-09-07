# Static-keys tenant extension

A Hindsight `TenantExtension` that authenticates requests with static API keys
declared in environment variables and gives every user their own PostgreSQL schema,
so memories are isolated at the database level.

- **Self-hosted, no identity provider**: users and keys come from the environment,
  no external IdP, no users table.
- **Schema per user**: a user `rafael` gets the schema `user_rafael` (lowercased,
  dashes become underscores), migrated on first access and cached afterwards.
- **Multiple keys per user**: `rafael:key1,rafael:key2` both authenticate as `rafael`
  into the same schema.
- **Constant-time key comparison** with `hmac.compare_digest` on every request.
- **Fail-fast on misconfiguration**: invalid entries, duplicate keys, schema-name
  collisions and over-long schema names are rejected at startup.

> This is a newer, dependency-free complement to
> [`supabase-tenant`](../supabase-tenant): where Supabase is the source of identity,
> this one is for fully self-hosted, single-node multi-user deployments with a
> handful of statically configured users.

## Install

Extensions are not published to PyPI. Build an image with this one in it, from the
repository root:

```bash
docker build -f hindsight-extensions/static-keys-tenant/Dockerfile -t hindsight-with-static-keys .
```

See the [Dockerfile](./Dockerfile) for what it does, and the
[packaging guide](../README.md#packaging-an-extension) for the general pattern.

To run the server outside Docker, put `hindsight_ext_static_keys_tenant/` on the
`PYTHONPATH` of the environment Hindsight runs in. There are no extra dependencies
to install — the extension only uses the server's own extension interfaces.

## Configure

```bash
HINDSIGHT_API_TENANT_EXTENSION=hindsight_ext_static_keys_tenant:StaticKeysTenantExtension
HINDSIGHT_API_TENANT_USERS=user1:key1,user1:key2,user2:key3
```

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `HINDSIGHT_API_TENANT_USERS` | yes | — | Comma-separated `user_id:api_key` pairs. Multiple keys may map to the same user |
| `HINDSIGHT_API_TENANT_SCHEMA_PREFIX` | no | `user` | Schema name prefix; must be a valid Postgres identifier |
| `HINDSIGHT_API_TENANT_MCP_AUTH_DISABLED` | no | `false` | Skip authentication for MCP endpoints (MCP clients land in the base schema) |

User IDs are **case-insensitive** and normalized before building the schema name:
they are lowercased and dashes become underscores (`Rafael`, `rafael` and `RAFAEL`
all resolve to `user_rafael`), matching how PostgreSQL folds unquoted identifiers.
Two distinct users whose ids collide after normalization (e.g. `jane-doe` vs
`jane_doe`, or ids longer than the 63-byte identifier limit) are rejected at startup.

Give the API and the worker the **same** variables: the worker calls `list_tenants()`
to decide which schemas to consolidate, so a worker without the extension leaves every
tenant's background processing stopped.

## Use

Clients pass their configured API key as a bearer token:

```bash
curl -H "Authorization: Bearer <key1>" \
  http://localhost:8888/v1/default/banks
```

Unknown or missing keys get a 401; every key is compared in constant time.

## Develop

```bash
uv sync
uv run pytest tests -v
```

## License

MIT.
