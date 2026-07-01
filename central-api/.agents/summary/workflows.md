# Workflows — CollabMind Control Plane

## 1. Memory Write (Agent → storage)

```mermaid
sequenceDiagram
    participant Agent
    participant Edge as collabmind-api (:3050)
    participant MC as memory-controller (:3020)
    participant GW as write-gate
    participant Adapter as internal adapter
    participant PG as Postgres

    Agent->>Edge: POST /api/memory/store\n{content, memory_type, ...}\nBearer mem11_sk_…
    Edge->>Edge: requireAuth() → resolve API key → RequestContext
    Edge->>Edge: signContext() → X-CM-Context header
    Edge->>MC: POST /memory/store + X-CM-Context
    MC->>MC: contextFromHeader() → verify HMAC
    MC->>GW: runWriteGate(content)
    GW->>GW: redactSecrets() — 8 patterns
    GW->>GW: classifySensitivity()
    GW->>GW: policyDecision()
    alt decision = reject
        MC-->>Agent: 422 rejected (secret content)
    else decision = quarantine
        MC->>Adapter: store(redacted, status=quarantined)
        Adapter->>PG: INSERT memories (state=quarantined)
        MC-->>Agent: 202 quarantined
    else decision = allow
        MC->>Adapter: store(redacted, status=active)
        Adapter->>PG: INSERT memories + memory_vectors
        MC-->>Agent: 201 created
    end
    MC->>PG: INSERT audit_events
```

## 2. Memory Search / Context Build (Agent retrieval)

```mermaid
sequenceDiagram
    participant Agent
    participant Edge as collabmind-api (:3050)
    participant MC as memory-controller (:3020)
    participant A1 as InternalAdapter
    participant A2 as OpenMemoryAdapter
    participant A3 as MemlordAdapter
    participant Unify as unifySearch (RRF)

    Agent->>Edge: POST /api/memory/search {query, k}
    Edge->>MC: proxy + X-CM-Context
    MC->>A1: search(query, k)
    MC->>A2: search(query, k)
    MC->>A3: search(query, k) [only if configured]
    A1-->>MC: hits[] (ranked)
    A2-->>MC: hits[] (ranked)
    A3-->>MC: hits[] (ranked)
    MC->>Unify: unifySearch([A1_hits, A2_hits, A3_hits], ctx, limit)
    Note over Unify: 1. metadata-first filter (status, confidentiality ceiling)
    Note over Unify: 2. RRF fusion RRF_K=60
    Note over Unify: 3. content-hash dedup (simhash / SHA-1)
    Note over Unify: 4. sort by fused_score desc
    Unify-->>MC: RankedHit[]
    MC->>MC: log audit_event (operation=search)
    MC-->>Agent: {results: RankedHit[], count}
```

### Context Pack Build (central-api Python contract)

Adds RRF across memory + code backends, builds a single cited bundle:

```mermaid
flowchart LR
    Q[query] --> MEM[memory adapters\nparallel search]
    Q --> CODE[code adapter\nsearch]
    MEM --> RRF[reciprocal_rank_fusion\nRRF_K=60]
    CODE --> RRF
    RRF --> DEDUP[content-key dedup\n200-char normalized prefix]
    DEDUP --> PACK[ContextPack\nselected_items, citations,\nblocked_count, confidence,\naudit_trace_id]
```

## 3. Document Ingestion (connector → Qdrant)

```mermaid
flowchart TD
    OP[Operator triggers sync\nvia /connectors/google-drive/sync] --> GDRIVE[Google Drive API\nfiles.list + files.get]
    GDRIVE --> REG[Register source_document\n+ source_document_permissions snapshot]
    REG --> JOB[Create source_ingestion_job\nstatus=pending]
    JOB --> DL[Download file content\nfrom Drive]
    DL --> PARSE[text parser\nchunker.ts]
    PARSE --> EMBED[embedding factory\nOllama / pgvector]
    EMBED --> QDRANT[(Qdrant\ncollection upsert)]
    EMBED --> PG[(Postgres\ndocument_chunks)]
    QDRANT --> DONE[job status=indexed]
    DL -->|error| FAIL[job status=failed\nerror logged]
```

Permission snapshots (`source_document_permissions`) are taken at sync time. Retrieval governance uses the snapshot, not live Drive ACL.

## 4. Agent Action Request (requires_approval flow)

```mermaid
sequenceDiagram
    participant Agent
    participant MC as central-api / MCP tools
    participant AC as agent_control.decide()
    participant Repo as repositories
    participant Console as collabmind-console

    Agent->>MC: request action (e.g. bulk_sync)
    MC->>AC: decide("bulk_sync")
    AC-->>MC: AgentDecision(requires_approval, "high_impact_action")
    MC->>Repo: record_agent_activity(decision=requires_approval)
    MC->>Repo: create_approval(pending)
    MC->>Repo: write_audit(AGENT_ACTION_REQUESTED)
    MC-->>Agent: {decision: "requires_approval", approval_id: "..."}
    Note over Console: Operator sees pending approval in /dashboard
    Console->>MC: POST /api/operator/approve {approval_id}
    MC->>Repo: update_approval(status=approved)
    MC->>Repo: write_audit(OPERATOR_APPROVED)
```

## 5. Retrieval Governance Check (per-document, fail-closed)

```mermaid
flowchart TD
    REQ[search_governed_documents call] --> MEMBER{workspace member?}
    MEMBER -->|no| DENY1[error: not_workspace_member]
    MEMBER -->|yes| LOOP[for each document]
    LOOP --> CONN{connector found?\nstatus=connected?}
    CONN -->|no| DENY2[deny: missing/inactive_connector]
    CONN -->|yes| DOC{document enabled?\nnot trashed?}
    DOC -->|no| DENY3[deny: document_disabled / trashed]
    DOC -->|yes| PERMS{permissions snapshot\nexists?}
    PERMS -->|null| DENY4[deny: unknown_permission]
    PERMS -->|exists| MATCH{any perm grants\nread to principal?}
    MATCH -->|no| DENY5[deny: not_permitted_on_source\naudit: RETRIEVAL_PERMISSION_DENIED]
    MATCH -->|yes| ALLOW[include in results]
```

Every denied document writes a `RETRIEVAL_PERMISSION_DENIED` audit event.

## 6. Auth Resolution Order

Both `collabmind-api` (edge) and `memory-controller` (internal) follow the same priority:

1. `x-cm-context` header → verify HMAC → trust (memory-controller only)
2. `X-Api-Key` header → SHA-256 lookup in `api_keys`
3. `Authorization: Bearer mem11_sk_…` → same as API key
4. `Authorization: Bearer eyJ…` → Authentik JWKS JWT RS256
5. Dev fallback (non-production only) → default tenant/actor

## 7. Operator Review Queue (console)

```mermaid
stateDiagram-v2
    [*] --> pending: Agent requests high-impact action
    pending --> approved: Operator approves via /api/operator/approve
    pending --> denied: Operator rejects via /api/operator/reject
    approved --> [*]: Action proceeds
    denied --> [*]: Action blocked, agent notified
```

Approval records live in `operator_approvals` table. All state changes are audited.
