# Data Models — CollabMind Control Plane

## RequestContext (shared concept, both repos)

The resolved identity attached to every request. Never derived from request body.

```python
# Python (central-api/app/auth/context.py)
class RequestContext(BaseModel):
    tenant_id: str
    actor_id: str
    workspace_id: str | None = None
    source_app_id: str | None = None
    roles: list[str]
    scopes: list[str]
    confidentiality_level: str = "internal"   # public|internal|private|sensitive|secret_blocked
    auth_method: Literal["jwt", "api_key", "dev_default"]
```

```typescript
// TypeScript (collabmind-api/src/auth.ts, memory-controller/src/middleware/auth.ts)
interface RequestContext {
  tenant_id: string
  actor_id: string
  user_id?: string
  source_app_id: string | null
  source_app_key: string | null
  roles: string[]
  scopes: string[]
  permissions: Record<string, unknown>
  auth_method: 'jwt' | 'api_key' | 'dev_default'
}
```

## AdapterHit

Normalized result from any memory backend:

```python
# Python
class AdapterHit(BaseModel):
    memory_id: str
    backend: str           # memlord|internal|openmemory|coderag
    content: str
    memory_type: str = "conversation"
    score: float | None    # raw backend score (not fused)
    citation: dict | None  # {backend, memory_id, workspace, file, line, chunk_id}
    metadata: dict         # backend-specific
```

## GovernanceResult / GateResult

Output of the write-gate:

```python
# Python
class GovernanceResult(BaseModel):
    content: str              # redacted version
    sensitivity: Sensitivity  # public|internal|private|sensitive|secret_blocked
    decision: PolicyDecision  # allow|quarantine|reject
    redactions: int
    reasons: list[str]
```

```typescript
// TypeScript
interface GateResult {
    content: string
    sensitivity: 'public' | 'internal' | 'restricted' | 'secret'
    decision: 'allow' | 'quarantine' | 'reject'
    redactions: number
    reasons: string[]
}
```

Note: Python uses `private/sensitive/secret_blocked`; TS uses `restricted/secret`. Semantically equivalent.

## ContextPack

Output of `build_context_pack()` — what agents receive as context:

```python
class ContextPack(BaseModel):
    query: str
    selected_items: list[ContextItem]
    citations: list[dict]
    blocked_items_count: int
    policy_decisions: list[dict]
    audit_trace_id: str
    confidence: float | None
    preview: bool = False

class ContextItem(BaseModel):
    backend: str
    content: str
    score: float
    citation: dict | None
```

## RankedHit (TS search unifier)

```typescript
interface RankedHit extends CollabMindMemory {
    fused_score: number
    sources: string[]           // which backends contributed
    citation: {
        backend: string
        backend_memory_id?: string
        file?: string
        line?: number
        chunk_id?: string
    }
}
```

## AuditEvent

```python
class AuditEvent(BaseModel):
    trace_id: str
    tenant_id: str
    actor_id: str | None
    operation: str        # store|search|update|delete|approve|reject|export|query|mcp_tool_called|agent_action_requested
    resource_type: str    # memory|context|connector|workspace|agent
    resource_id: str | None
    outcome: str          # success|error|forbidden|denied
    metadata: dict
    timestamp: int        # ms since epoch
```

## DB Tables (central-api, SQLAlchemy Core)

### users
| Column | Type | Notes |
|--------|------|-------|
| id | String(36) PK | UUID |
| email | String(320) UNIQUE | |
| display_name | String(200) | |
| is_operator | Boolean | |
| created_at | DateTime(tz) | |

### workspaces
| Column | Type |
|--------|------|
| id | String(36) PK |
| name | String(200) |
| owner_id | FK → users.id |
| created_at | DateTime(tz) |

### workspace_members
| Column | Type | Notes |
|--------|------|-------|
| id | String(36) PK | |
| workspace_id | FK → workspaces.id | |
| user_id | FK → users.id | |
| role | String(32) | owner\|admin\|member |
| created_at | DateTime(tz) | |
| — | UNIQUE(workspace_id, user_id) | |

### source_connectors
| Column | Type | Notes |
|--------|------|-------|
| id | String(36) PK | |
| workspace_id | FK → workspaces.id | |
| provider | String(64) | "google_drive" |
| status | String(32) | disconnected\|connected\|error |
| connected_by | FK → users.id | |
| account_email | String(320) | |
| config | JSON | e.g. `{"folder_ids": [...]}` |
| created_at, updated_at | DateTime(tz) | |

### source_documents
| Column | Type | Notes |
|--------|------|-------|
| id | String(36) PK | |
| workspace_id | FK → workspaces.id | |
| connector_id | FK → source_connectors.id | |
| provider | String(64) | |
| external_id | String(256) | drive_file_id |
| name, mime_type, size, web_view_link | various | |
| checksum | String(128) | change detection |
| trashed, enabled | Boolean | `enabled=false` → excluded from retrieval |
| sync_status | String(32) | discovered\|downloading\|parsing\|chunking\|embedding\|indexed\|skipped\|failed |
| metadata | JSON | owners, parents, timestamps |

### source_document_permissions
| Column | Type | Notes |
|--------|------|-------|
| document_id | FK → source_documents.id | |
| ptype | String(32) | user\|group\|domain\|anyone |
| role | String(32) | owner\|organizer\|writer\|commenter\|reader |
| email_address, domain | String | |
| allow_file_discovery | Boolean | |
| expiration_time | DateTime(tz) | checked on retrieval |

### agent_activity
| Column | Type | Notes |
|--------|------|-------|
| agent_id | String(128) | |
| requested_action | String(128) | |
| decision | String(32) | allowed\|denied\|requires_approval |
| reason | Text | |

### operator_approvals
| Column | Type | Notes |
|--------|------|-------|
| requested_by | String(128) | agent_id or user_id |
| action | String(128) | |
| status | String(32) | pending\|approved\|denied |
| decided_by | String(36) | FK → users.id |
| decided_at | DateTime(tz) | |

## Identity Hierarchy (collabmind-memory/core)

```
tenant (billing/isolation boundary)
  └─ actor (human | agent | tool | system)
      └─ source_app (client application)
          └─ project (workspace / environment)
              └─ memory | temporal_fact
```

All `api_keys` store `SHA-256(key)` hash — raw key never persisted. `expires_at` is ms since epoch (NULL = no expiry).

## Sensitivity Levels (both repos, same ordering)

| Level | Python enum | TS type | Write decision | Retrieval |
|-------|------------|---------|---------------|-----------|
| public | `Sensitivity.public` | `'public'` | allow | yes |
| internal | `Sensitivity.internal` | `'internal'` | allow | yes (within ceiling) |
| private / restricted | `Sensitivity.private` | `'restricted'` | quarantine | no (pending review) |
| sensitive | `Sensitivity.sensitive` | — | quarantine | no |
| secret_blocked | `Sensitivity.secret_blocked` | `'secret'` | **reject** | no |
