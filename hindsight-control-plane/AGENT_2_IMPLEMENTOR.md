# CollabMind Brain — Implementor Agent

**Role:** Builder & Integrator  
**Purpose:** Write code, port components from the old stack, integrate frontend ↔ backend, and ship operator-owned intelligence.  
**Base:** Branch `feat/collabmind-brain`, repo `collabmind-hindsight`  

---

## Before Writing Any Code

1. **Read the Architect Agent first** (`AGENT_1_ARCHITECT.md`) — every implementation must respect the Core Identity, Four Pillars, Anti-Drift Rules, and Brand Language.
2. **Read the reference implementation** — check `XAI-v2/stacks/collabmind/` for existing patterns before writing anything new.
3. **Verify service state** — never assume a port, model, or endpoint exists. Query the Backplane.
4. **Check brand compliance** — does this UI show state, source, permission, and verification? Hidden automation is against the brand.
5. **Read existing files before modifying** — never assume.

---

## Visual Implementation Guide

### Color Palette (Provisional — From Existing Assets)

```css
/* Dark Operator Cockpit Theme */
--cm-primary-dark: #222833;     /* Page backgrounds */
--cm-deep-slate: #2F3A43;       /* Cards, panels */
--cm-steel-slate: #3C4C53;      /* Borders, dividers */
--cm-muted-teal: #46686F;       /* Secondary elements */
--cm-system-teal: #61898F;      /* Hover, active states */
--cm-soft-grey: #9FAFB3;        /* Secondary text */
--cm-pale-signal: #DFECEC;      /* Subtle highlights */
--cm-electric-cyan: #43D1D3;    /* Primary accent, active indicators */

/* Semantic status colors */
--cm-status-green: #22C55E;     /* Running, approved, healthy */
--cm-status-amber: #F59E0B;     /* Warning, pending, needs review */
--cm-status-red: #EF4444;       /* Blocked, error, rejected */
--cm-status-blue: #3B82F6;      /* Info, processing */
```

### Tailwind Equivalents
```tsx
// Currently available classes in the control plane
bg-background         → --cm-primary-dark
bg-card               → --cm-deep-slate
bg-muted              → --cm-steel-slate
text-foreground       → white / near-white
text-muted-foreground → --cm-soft-grey
text-primary          → --cm-electric-cyan
border-border         → --cm-steel-slate
```

### The Design System Screens

Every UI you build maps to one of these archetypes:

| Screen | Purpose |
|--------|---------|
| **Operator Dashboard** | Active tasks, system health, pending approvals, risk state |
| **Run Timeline** | Request → plan → retrieval → tool call → diff → approval → result → verification |
| **Agent Registry** | Agent role, model, permissions, last action, trust score, failure history |
| **Model Registry** | Model type, endpoint, modality, context, capabilities (tools/images/audio/embeddings), verified test result |
| **Backplane View** | Ports, services, containers, databases, parsers, vector stores, queues, model servers |
| **Memory Console** | Search, approve, reject, quarantine, supersede, redact, delete, export |
| **Evaluation Console** | Datasets, test cases, scores, failure categories, regression results |
| **Vector Explorer** | Chunks, payloads, scores, sources, metadata filters |
| **Policy Center** | Prime directives, approval rules, scope boundaries, secrets policy |

---

## Code Patterns

### Console Page Pattern

Every page in the reference console follows this structure:

```tsx
'use client';
import { useCallback, useEffect, useState } from 'react';
import { api } from '@/lib/api';

type SomeResponse = { ... };

export default function Page() {
  const [data, setData] = useState<...>([]);
  const [err, setErr] = useState('');
  const [loading, setLoading] = useState(true);

  const loadAll = useCallback(() => {
    setLoading(true);
    api.get<SomeResponse>('/api/some/endpoint')
      .then(d => { setData(d.items); setErr(''); })
      .catch(() => setErr('Description of error'))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { loadAll(); }, [loadAll]);

  return (
    <>
      <div className="topbar">
        <span className="topbar-title">Page Title</span>
      </div>
      <div className="page">
        <h1>Page Title</h1>
        {err && <div className="error-msg">{err}</div>}
        <div className="card">
          <div className="card-label">Section Title</div>
          {loading ? (
            <div className="dim">Loading…</div>
          ) : items.length === 0 ? (
            <div className="nodata-msg">No data</div>
          ) : (
            <table>...</table>
          )}
        </div>
      </div>
    </>
  );
}
```

### Brand-Compliant UI Checklist
- [ ] Shows current state (loading / empty / data / error)
- [ ] Shows source of data (which service, endpoint, or adapter)
- [ ] Shows permission level if applicable (read-only, requires approval, etc.)
- [ ] Shows verification status if applicable (verified, pending, failed)
- [ ] No hidden automation — if something happens asynchronously, show a progress indicator
- [ ] Uses the palette: dark background, teal/cyan accents, clear typography

### API Route Pattern

```python
from fastapi import APIRouter, Depends
from app.auth import require_operator

router = APIRouter(prefix="/api/some", tags=["some"])

@router.get("/items")
async def list_items(operator=Depends(require_operator)):
    """Description of what this does."""
    items = await service.list_all(tenant_id=operator.tenant_id)
    return {"items": items, "count": len(items)}
```

### Core Infrastructure Patterns (From Old Stack)

**Verify-Once-at-Edge**: One token verifier at the edge. After auth, serialize `RequestContext`, sign as `X-CM-Context`, forward to services. No downstream service re-verifies the token.

**Write-Gate**: Every write runs: `redactSecrets()` → `classifySensitivity()` → `policyDecision()`
- `secret_blocked` → **reject** (not stored)
- `private` / `sensitive` → **quarantine** (stored, excluded from retrieval)
- `public` / `internal` → **allow**

**Fail-Closed Retrieval**: `evaluate_retrieval()` denies on any unknown. Missing connector, missing snapshot, trashed document all produce deny + audit. Never return partial results with gaps.

**Tenant Isolation**: `tenant_id` is always from `req.context`. Never from request body or query string.

**RRF Constant**: `RRF_K = 60` in both repos. Change both if you change either.

**No Secrets in Responses**: Provider registry omits `api_key`. Settings null `value_json`. Models have no auth material.

---

## Reference Code Locations

### Console Pages (Port from old CollabMind)

| Old Path | Purpose | Port Target |
|----------|---------|-------------|
| `console/src/app/models/page.tsx` | Model Center — provider registry + model inventory | `hindsight-control-plane/src/app/[locale]/models/` |
| `console/src/app/router/page.tsx` | Router — route preview + decision history | `hindsight-control-plane/src/app/[locale]/router/` |
| `console/src/app/mcp/page.tsx` | MCP Tools — tool registry table | `hindsight-control-plane/src/app/[locale]/mcp/` |
| `console/src/app/api-center/page.tsx` | API Center — service cards + route table | `hindsight-control-plane/src/app/[locale]/api-center/` |
| `console/src/app/agents/page.tsx` | Agent Registry | `hindsight-control-plane/src/app/[locale]/agents/` |
| `console/src/app/governance/page.tsx` | Policy Center — rules, quarantine, approvals | `hindsight-control-plane/src/app/[locale]/governance/` |
| `console/src/app/audit/page.tsx` | Audit log viewer | `hindsight-control-plane/src/app/[locale]/audit/` |
| `console/src/app/connectors/page.tsx` | Connector management | `hindsight-control-plane/src/app/[locale]/connectors/` |
| `console/src/app/chat/page.tsx` | Chat playground | `hindsight-control-plane/src/app/[locale]/chat/` |
| `console/src/app/health/page.tsx` | Health monitoring | `hindsight-control-plane/src/app/[locale]/health/` |
| `console/src/app/settings/page.tsx` | Configuration | `hindsight-control-plane/src/app/[locale]/settings/` |
| `console/src/app/dashboard/page.tsx` | Operator Dashboard landing | `hindsight-control-plane/src/app/[locale]/dashboard/` |
| `console/src/app/memory/page.tsx` | Memory Console | `hindsight-control-plane/src/app/[locale]/memory/` |
| `console/src/app/playground/page.tsx` | AI Playground | `hindsight-control-plane/src/app/[locale]/playground/` |

### Central API Modules (Port to new backend)

| Old Path | Purpose | CollabMind Component |
|----------|---------|----------------------|
| `central-api/app/ai/` | AI Gateway — providers, models, router | Cognitive Router + Model Registry |
| `central-api/app/mcp/` | MCP tool registry + gateway | Tool Registry |
| `central-api/app/governance/` | Write-gate, quarantine, approvals | Policy + Approval Gateway |
| `central-api/app/audit/` | Audit logging | Observability / Provenance / Audit |
| `central-api/app/auth/` | JWT + API key auth | Auth layer (Part of Constitution) |
| `central-api/app/connectors/` | Source connectors | Tool Registry (connector adapters) |
| `central-api/app/ingestion/` | Document ingestion | Memory / Knowledge Plane |
| `central-api/app/retrieval/` | Retrieval pipeline | Cognitive Router (retrieval leg) |
| `central-api/app/operator/` | Operator backend | Operator Interface backend |
| `central-api/app/controlplane/` | Control plane | Control Plane |

### API Edge (Port from old Express)

| Old Path | Purpose | CollabMind Component |
|----------|---------|----------------------|
| `api/src/auth.ts` | JWT + API key verification | Constitution enforcement point |
| `api/src/proxy.ts` | Proxy with X-CM-Context signing | Verification boundary |
| `api/src/internal-context.ts` | Context signing | Tenant isolation enforcement |
| `api/src/middleware/` | Auth middleware | Policy enforcement middleware |
| `api/src/routes/` | Route handlers | Inspect for patterns |

---

## Implementation Workflow

### Step 1: Read Reference
```bash
# Before building anything new, check if it exists
ls /Users/oliververmeulen/XAI-v2/stacks/collabmind/console/src/app/<page>/
ls /Users/oliververmeulen/XAI-v2/stacks/collabmind/central-api/app/<module>/
```

### Step 2: Port Pattern, Don't Copy Blindly
- Understand the old implementation
- Adapt to the current Hindsight Control Plane architecture
- Map to the correct CollabMind component (see tables above)
- Use existing UI components (Card, Badge, Table, Button, etc.)
- Follow the i18n pattern from `en.json`

### Step 3: Implement in Order
1. **Backend API first** (route + service)
2. **Frontend component** (page + sub-components)
3. **i18n strings** (all 10 locales)
4. **TypeScript types** (shared types in `lib/`)
5. **Brand compliance check** — see checklist above
6. **Test build** (`npm run build`)

### Step 4: Verify
- `npm run build` passes
- Page loads without errors
- All API calls succeed
- i18n keys exist for all 10 locales
- Brand-compliant: state, source, permission, verification all visible

---

## Current UI Component Library

Available at `hindsight-control-plane/src/components/ui/`:
- `card.tsx` — Card, CardHeader, CardContent, CardTitle, CardDescription
- `badge.tsx` — Badge with variants
- `button.tsx` — Button with variants
- `table.tsx` — Table with all sub-components
- `tabs.tsx` — Tabs, TabsList, TabsTrigger, TabsContent
- `switch.tsx` — Switch toggle
- `select.tsx` — Select dropdown
- `dialog.tsx` — Modal dialog
- `alert-dialog.tsx` — Confirmation dialogs
- `dropdown-menu.tsx` — Context menus
- `sonner.tsx` — Toast notifications

Available at `hindsight-control-plane/src/components/`:
- `sidebar.tsx` — NavItem: data|recall|reflect|documents|entities|uploads|profile
- `bank-selector.tsx` — Bank dropdown
- `pending-uploads-view.tsx` — Polling operations viewer
- `deployment-view.tsx` — Config cards
- `search-debug-view.tsx` — RRF trace visualization
- `brand-style-injector.tsx` — CSS var injection

Icons: All `lucide-react` icons.

---

## i18n Pattern

All locale files in `hindsight-control-plane/src/messages/`. Ten locales: `en`, `de`, `fr`, `ja`, `ko`, `pt`, `es`, `zh-CN`, `zh-TW`, `yue-Hant`.

Adding a new section:
1. Edit `en.json` first (authoritative English)
2. Propagate to all 9 other locales:
```python
import json, os
locales = ['de', 'fr', 'ja', 'ko', 'pt', 'es', 'zh-CN', 'zh-TW', 'yue-Hant']
for loc in locales:
    path = os.path.join('src/messages', f'{loc}.json')
    with open(path) as f: data = json.load(f)
    data['newSection'] = { 'key': 'value' }
    with open(path, 'w') as f: json.dump(data, f, indent=2, ensure_ascii=False); f.write('\n')
```

---

## Git Workflow

```bash
git checkout feat/collabmind-brain
git checkout -b feat/collabmind-brain/<module-name>

git commit -m "feat(<module>): <description>"

git rebase feat/collabmind-brain
```

---

## Build & Test Commands

```bash
cd hindsight-control-plane && npm run build     # Full build
cd hindsight-control-plane && npm run dev       # Dev server
cd hindsight-control-plane && npx tsc --noEmit  # Type check only
```

---

## Anti-Drift Rules (Hard Rules)

1. **Do not rename CollabMind** unless the operator explicitly asks.
2. **Do not treat infrastructure tools as the product.** They are adapters. CollabMind is the governance layer above them.
3. **Do not invent capabilities.** Verify model, parser, endpoint, service, and port state first.
4. **Do not store memory automatically** unless policy allows it or the operator commands it.
5. **Do not use raw logs, secrets, credentials, wallet data, private keys, or temporary debug junk as memory.**
6. **Do not overfit the brand to telecom.** Telecom/audio is the first vertical, not the whole platform.
7. **Do not let agent language override operator language.** Use "operator," "control plane," "cognitive router," "backplane," "memory governance."
8. **Do not claim improvement without trace + evaluation + score.**
9. **Do not assume local means safe.** Local-first still needs governance, auth, scope filtering, redaction, and audit.
10. **Every UI must show state, source, permission, and verification.** Hidden automation is against the brand.
11. **No secrets in responses** — never expose api_key, tokens, or credentials.
12. **i18n is not optional** — every user-facing string must be in all 10 locales.
