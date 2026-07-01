# Interfaces — CollabMind Control Plane

## External API Surface (collabmind-api :3050)

All routes require `Authorization: Bearer <jwt>` or `X-Api-Key: mem11_sk_…` except `/health`.

### Memory

| Method | Path | Upstream | Notes |
|--------|------|----------|-------|
| `*` | `/api/memory/**` | `memory-controller:3020/memory/**` | Full proxy with X-CM-Context. All methods/subpaths forwarded. |

### Documents

| Method | Path | Upstream |
|--------|------|----------|
| `*` | `/api/documents/**` | `memory-controller:3020/documents/**` |

### Settings / Secrets / Services

Direct Postgres — not proxied:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/settings` | Query `platform_settings`. Encrypted rows: `value_json` nulled. |
| `POST` | `/api/settings` | Create setting. |
| `PUT` | `/api/settings/:id` | Update setting. |
| `DELETE` | `/api/settings/:id` | Delete setting. |
| `GET` | `/api/secrets` | List secrets (refs only, no values). |
| `POST` | `/api/secrets` | Create secret. |
| `GET` | `/api/services` | List service configs. |

### Graph / Evaluation / Model / Admin / MCP

| Path | Upstream |
|------|----------|
| `/api/graph/**` | proxied |
| `/api/evaluation/**` | proxied |
| `/api/model/**` | proxied |
| `/api/admin/**` | proxied |
| `/mcp/**` | proxied |

### Auth / Health

| Path | Auth required |
|------|--------------|
| `GET /health` | No |
| `GET /auth/**` | No |

---

## central-api Contract (:8000)

The Python scaffold defines the final API contract:

### Memory

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/memory/search` | Search across all configured adapters. |
| `POST` | `/api/memory/store` | Write-gate → store → audit. |
| `POST` | `/api/memory/update` | Update existing memory. |
| `POST` | `/api/memory/delete` | Soft-delete. |
| `POST` | `/api/memory/export` | Export all (paginated from MemLord + internal). |

### Retrieval

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/retrieval/code/search` | Code search via CodeRAG adapter. |
| `POST` | `/api/retrieval/docs/search` | Docs search (CodeRAG covers both in scaffold). |
| `POST` | `/api/context/build` | Multi-backend RRF fusion → context-pack. Audited. |
| `POST` | `/api/context/preview` | Same as build, no audit write, preview flag set. |

### Operator

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/operator/review` | List pending approvals. |
| `POST` | `/api/operator/approve` | Approve a queued action. |
| `POST` | `/api/operator/reject` | Reject a queued action. |

### Control Plane

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/controlplane/users` | List users. |
| `POST` | `/controlplane/users` | Create user. |
| `GET` | `/controlplane/workspaces` | List workspaces. |
| `POST` | `/controlplane/workspaces` | Create workspace. |
| `POST` | `/controlplane/workspaces/{id}/members` | Add member. |

### Connectors

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/connectors` | List connectors for workspace. |
| `GET` | `/connectors/{id}/status` | Connector status. |
| `GET` | `/connectors/google-drive/oauth-config` | OAuth status (no secrets). |
| `GET` | `/connectors/google-drive/oauth` | Start OAuth flow. |
| `GET` | `/connectors/google-drive/callback` | OAuth callback. |
| `POST` | `/connectors/google-drive/connect` | Connect Drive to workspace. |
| `POST` | `/connectors/google-drive/sync` | Trigger manual sync. |

### Audit

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/audit/events` | List audit events for tenant. |

### Health

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check. |
| `GET` | `/api/health/engines` | Health of all registered adapters. |

---

## MCP Tools (central-api / mcp-server)

All 7 tools require workspace membership and emit a `mcp_tool_called` audit event:

| Tool | Parameters | Returns |
|------|-----------|---------|
| `list_workspaces` | `actor_id` | `{workspaces: [...]}` |
| `list_connected_sources` | `actor_id`, `workspace_id` | `{connectors: [...]}` |
| `list_source_documents` | `actor_id`, `workspace_id` | `{documents: [...]}` |
| `sync_source` | `actor_id`, `workspace_id` | sync result |
| `get_source_audit` | `actor_id`, `workspace_id`, `limit` | `{events: [...]}` |
| `list_ingestion_jobs` | `actor_id`, `workspace_id` | `{jobs: [...]}` |
| `search_governed_documents` | `actor_id`, `workspace_id`, `email?`, `domain?` | `{results: [...], denied_count: N}` |

`search_governed_documents` applies full fail-closed retrieval governance per document. Denied items are audited individually.

---

## Internal Service Interface: X-CM-Context

Header name: `x-cm-context`

Format: `<base64url(json)>.<base64url(hmac_sha256(json, INTERNAL_CONTEXT_SECRET))>`

Payload (`InternalContext`):
```typescript
{
  tenant_id: string
  actor_id: string
  source_app_id?: string | null
  user_id?: string
  roles: string[]
  scopes: string[]
  auth_method: 'jwt' | 'api_key' | 'dev_default'
}
```

Internal services (`memory-controller`) call `contextFromHeader()` / `verifyContext()` to obtain the resolved identity. If the header is absent or HMAC fails → fall through to other auth methods.

---

## Adapter Interface (BaseMemoryAdapter / AdapterInterface)

### Python (central-api)

```python
class BaseMemoryAdapter(ABC):
    backend: str        # identifier
    read_only: bool     # if True, store/update/delete raise NotImplementedError

    async def search(query, *, workspace_id, memory_type, k) -> list[AdapterHit]: ...
    async def store(content, *, memory_type, workspace_id, tags, metadata) -> AdapterHit: ...
    async def update(memory_id, content, metadata) -> AdapterHit: ...
    async def delete(memory_id) -> None: ...
    async def export(workspace_id) -> list[dict]: ...
    async def health() -> EngineHealth: ...
```

### TypeScript (memory-controller)

```typescript
interface MemoryAdapter {
    search(query: string, ctx: RequestContext, opts: SearchOpts): Promise<AdapterHit[]>
    store(content: string, ctx: RequestContext, opts: StoreOpts): Promise<AdapterHit>
    // read-only adapters throw on store/update/delete
}
```

---

## Auth Flow Summary

```mermaid
flowchart TD
    REQ[Incoming request] --> CHECK{Header?}
    CHECK -->|"Authorization: Bearer eyJ..."| JWT[Verify JWT\nAuthentik JWKS RS256]
    CHECK -->|"Authorization: Bearer mem11_sk_…"| APIKEY[Look up key_hash\nin api_keys table]
    CHECK -->|"X-Api-Key: mem11_sk_…"| APIKEY
    CHECK -->|"x-cm-context header"| HMAC[Verify HMAC\ntrust context]
    JWT -->|valid| CTX[Build RequestContext]
    APIKEY -->|found + not expired| CTX
    HMAC -->|valid| CTX
    CTX --> NEXT[next()]
    JWT -->|invalid| DEV{IS_DEV?}
    APIKEY -->|not found| DEV
    DEV -->|yes| DEFCTX[dev default context\ntenant=00000001 actor=00000002]
    DEV -->|no| 401[401 unauthorized]
```
