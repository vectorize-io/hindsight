# CollabMind Brain — Architecture & Plan

> **Working Name**: CollabMind Brain
> 
> **The Vision**: An open-source AI infrastructure operating system. Not just a memory viewer — a unified command center for memory, agents, compute, storage, cloud edge, and AI — all from one panel, all scriptable, all with an AI co-pilot.

---

## Current State

Three separate surfaces that don't talk to each other:

| Surface | Port | Role |
|---------|------|------|
| **Control Plane** (Next.js) | 9998 | Bank management, search debug, system monitor |
| **Embed Control Center** | 7878 | Lightweight all-in-one appliance UI |
| **Memlord MCP** | 8005 | Agent memory server (separate from Hindsight) |

**Problem**: User has to context-switch between three UIs. No unified view of the full stack.

---

## Target Architecture: Unified Admin Panel

```
┌──────────────────────────────────────────────────────────────────┐
│                   COLLABMIND BRAIN                                 │
│              Unified AI Infrastructure Admin Panel                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┐ ┌───────────┐ ┌───────────┐ ┌──────────┐          │
│  │ MEMORY   │ │ MCP       │ │ CLOUD-    │ │ IAM      │          │
│  │ OPERATIONS│ │ MANAGER   │ │ FLARE     │ │ USERS    │          │
│  ├──────────┤ ├───────────┤ ├───────────┤ ├──────────┤          │
│  │ Banks    │ │ Server    │ │ Workers   │ │ Roles    │          │
│  │ Recall   │ │ Registry  │ │ Durable   │ │ Access   │          │
│  │ Retain   │ │ Tools     │ │ Objects   │ │ API Keys │          │
│  │ Reflect  │ │ Explorer  │ │ R2 / KV   │ │ Audit    │          │
│  │ Graph    │ │ Perms     │ │ D1 / AI   │ │ SSO      │          │
│  │ Docs     │ │ Audit     │ │ Gateway   │ │          │          │
│  └──────────┘ └───────────┘ └───────────┘ └──────────┘          │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              CLAUDE AI CO-PILOT                           │    │
│  │  Natural language infrastructure control                 │    │
│  │  ─────────────────────────────────────────────             │    │
│  │  "Scale to 8 workers" → Hindsight API                    │    │
│  │  "Deploy DO for bank sync" → Cloudflare API              │    │
│  │  "Show failed ops last hour" → Audit Logs                │    │
│  │  "Add user Alice with admin role" → IAM API              │    │
│  │  "Backup all memories to R2" → Orchestrated pipeline     │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ════════════════════════════════════════════════════════════    │
│  MCP Protocol Layer — Every capability exposed as MCP tools       │
│  Any MCP-compatible agent (Claude, ChatGPT, Cursor) can           │
│  control infrastructure through the same interface.               │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Core Modules

### 1. Memory Operations (Existing — Enhance)

**What it does now**: Bank CRUD, recall/reflect, search debug, documents, entities, mental models

**Enhancements**:
- Real-time memory graph with interactive exploration
- Batch operations (tag all, delete all, export/import across banks)
- Memory diff viewer (show what changed between consolidations)
- Scheduled operations (nightly consolidation, weekly mental model refresh)
- Memory quality scoring (completeness, consistency, staleness)

### 2. MCP Manager (New — Phase A)

Manage every MCP server from the dashboard:

- **Server Registry**: List all connected MCP servers, their status, version
- **Tool Explorer**: Browse available tools per server, search by name/description
- **Live Monitor**: Real-time MCP call volume, latency, error rates per server
- **Permission Matrix**: "Which agent can use which tools on which banks?"
- **Health Dashboard**: Connection status, last heartbeat, restart button
- **Logs**: Per-server MCP call logs with filtering

**How**: The MCP protocol is standardized. Build an orchestrator service that manages MCP server processes and exposes their metadata to the UI.

### 3. Cloudflare Integration (New — Phase B)

Manage Cloudflare infrastructure from the panel:

- **Workers**: Deploy, update, rollback, view logs, set environment variables
- **Durable Objects**: Create/manage namespaces, inspect storage, reset state
- **R2 Storage**: Browse buckets, set policies, generate presigned URLs, sync
- **KV**: Query namespaces, edit values, bulk import/export
- **D1 Databases**: Run queries, view schema, run migrations
- **AI Gateway**: Route LLM calls, view caching stats, rate limits
- **Tunnels**: Manage `cloudflared` config, view tunnel status
- **Edge Workers**: Deploy lightweight Hindsight workers to the edge

**How**: Uses Cloudflare API client in the backend. Each resource type gets a CRUD API endpoint + UI page.

### 4. Durable Storage — Classmate Workers (Phase C)

Replace or augment local storage with Durable Objects + R2:

```
┌─────────────┐    ┌──────────────────────┐    ┌────────────┐
│  Local      │◄──►│  Durable Objects     │◄──►│  Cloud     │
│  Hindsight  │    │  (state + coord)     │    │  Workers   │
│  (pg0)      │    │  ┌────────────────┐  │    │  (compute) │
│             │    │  │ KV: config     │  │    │            │
│  Embed      │    │  │ R2: blob store │  │    │  Edge AI   │
│  Control    │    │  │ DO: locks,     │  │    │  Gateway   │
│  Center     │    │  │     coordination│  │    │            │
│             │    │  └────────────────┘  │    │  Global    │
└─────────────┘    └──────────────────────┘    │  Deploy    │
                                                └────────────┘
```

**Why**: Removes dependency on local PostgreSQL for embedded deployments. Workers can run anywhere — no local state required.

### 5. IAM & User Management (New — Phase E)

Multi-tenant access control:

- **Users**: Invite, manage, suspend
- **Roles**: Admin, Operator, Viewer, custom
- **Permissions**: Per-bank, per-tool, per-feature
- **API Keys**: Generate, scope, rotate, revoke
- **Audit Log**: Every action logged with user, timestamp, resource
- **SSO**: OAuth2/OIDC integration (Google, GitHub, Cloudflare)

**Required for**: Any multi-tenant deployment or organizational use.

### 6. Claude AI Co-Pilot (New — Phase D)

An embedded AI assistant in the admin panel:

- **Natural language interface** for all operations
- **Context-aware**: Knows which bank/user/session you're in
- **Multi-step**: "Deploy 4 workers with Durable Objects for bank X"
- **Feedback loop**: Executes, reports success/failure, suggests next steps
- **Conversation history**: Remembers previous commands

**Technical approach**: Uses Hindsight's own Reflect engine with a bank configured for infrastructure operations. The Co-Pilot is itself a Hindsight agent.

---

## Backend Architecture

```
┌────────────┐    ┌──────────────────┐    ┌─────────────┐
│  Frontend  │◄──►│  Orchestrator    │◄──►│  Cloudflare │
│  (Next.js) │    │  API Service     │    │  API        │
│            │    │  (Python/FastAPI) │    │             │
│  Unified   │    │                  │    │  Workers    │
│  Admin UI  │    │  ┌────────────┐  │    │  DO         │
│            │    │  │ MCP Proxy  │  │    │  R2/KV/D1  │
│  Embed     │    │  │ Cloudflare │  │    │  AI Gateway│
│  Control   │    │  │ Client     │  │    │  Tunnels   │
│  Center    │    │  │ IAM Engine │  │    │             │
│            │    │  │ Claude     │  │    └─────────────┘
│            │    │  │ Co-Pilot   │  │
│            │    │  └────────────┘  │
│            │    │                  │
│            │    │  ┌────────────┐  │    ┌─────────────┐
│            │    │  │ Hindsight  │  │    │  Memlord    │
│            │    │  │ Engine     │◄─┤───►│  MCP        │
│            │    │  │ (memory)   │  │    │  Server     │
│            │    │  └────────────┘  │    └─────────────┘
└────────────┘    └──────────────────┘
```

### Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Frontend | Next.js (existing) | Reuse existing Control Plane codebase |
| Backend API | Python/FastAPI | Hindsight already Python; MCP and Cloudflare SDKs available |
| MCP Layer | Standard MCP protocol | Industry standard, compatible with all major AI agents |
| Auth | OIDC + JWT | SSO-ready, stateless, works with Cloudflare Access |
| State | PostgreSQL + Durable Objects | Hybrid — local for low latency, cloud for durability/sharing |
| Deployment | Embed + Cloudflare | Single binary for local, edge-deployable to Workers |

---

## Phased Build Plan

### Phase A — MCP Manager (2-3 weeks)

**Goal**: See and control all MCP servers from the admin panel.

- [ ] Backend: MCP orchestrator service — process management, health checks, tool discovery
- [ ] Backend: MCP permission model — which tools for which bank/agent
- [ ] Backend: MCP audit logging
- [ ] Frontend: MCP Server Registry page (list, status, restart)
- [ ] Frontend: Tool Explorer (browse + search tools)
- [ ] Frontend: Permission Matrix UI (bank × tool × agent)

### Phase B — Cloudflare Shell (3-4 weeks)

**Goal**: Deploy and manage Workers/DO from the panel.

- [ ] Backend: Cloudflare API client wrapper (auth, rate limiting, error handling)
- [ ] Backend: Worker CRUD — deploy, update, rollback, get logs
- [ ] Backend: DO CRUD — namespace management, storage inspection
- [ ] Backend: R2 browser — list buckets, objects, generate URLs
- [ ] Backend: KV/D1 read-only explorer
- [ ] Frontend: Cloudflare section in admin panel
- [ ] Frontend: Worker editor (code + env vars + triggers)
- [ ] Frontend: R2 file browser UI

### Phase C — Durable Storage Backend (4-6 weeks)

**Goal**: Hindsight workers can run fully on Cloudflare infrastructure.

- [ ] Durable Objects as worker coordination layer
- [ ] R2 as blob storage backend (replaces local fs)
- [ ] KV for config/service discovery
- [ ] DO-backed job queue (replaces pg0-based queue)
- [ ] Hybrid mode: local pg0 + cloud DO sync
- [ ] Migration tool: pg0 → Cloudflare storage

### Phase D — Claude Co-Pilot (2-3 weeks)

**Goal**: Natural language control of the entire platform.

- [ ] New Hindsight bank: `collabmind-brain` with infrastructure knowledge
- [ ] MCP tools exposing all admin API endpoints
- [ ] Chat UI in the admin panel (sidebar or modal)
- [ ] Multi-step orchestration (reflection loop)
- [ ] Confirmation dialogs for destructive operations

### Phase E — IAM & Multi-Tenant (2-3 weeks)

**Goal**: Production-grade access control.

- [ ] User model + database schema
- [ ] OIDC/OAuth2 login flow
- [ ] Role-based access control (RBAC)
- [ ] API key management with scopes
- [ ] Audit log viewer in admin panel
- [ ] Session management

### Phase F — Unification (1-2 weeks)

**Goal**: One panel to rule them all.

- [ ] Merge Control Plane (9998) into Embed Control Center (7878)
- [ ] Migrate all pages to the unified shell
- [ ] Remove the separate Next.js dev server in production
- [ ] Single binary: `hindsight-embed serve` = everything

---

## Quick Start (Branch Setup)

```bash
git checkout -b feat/collabmind-brain
# This doc lives at:
# hindsight-control-plane/COLLABMIND_BRAIN_ARCHITECTURE.md
```

---

## Open Questions

1. **Branding**: "CollabMind Brain" vs "CollabMind Mind" vs something else?
2. **License**: Keep open-source (MIT/APACHE) with enterprise tiers?
3. **CLI-first**: Should every UI action also be available as a CLI command?
4. **Plugin system**: Should modules (MCP Manager, Cloudflare, IAM) be plugins loaded at runtime?
5. **Self-hosted vs SaaS**: Pure self-hosted, or offer a managed Cloudflare version?

---

*This is a living document. Update as the vision evolves.*
