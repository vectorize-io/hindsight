# UI Integrations — Feature Summary

**Branch**: `feat/ui-integrations`  
**Commit**: `a558f4a26`  
**Date**: 2026-06-28  

## Overview

Five UI integration features for the Hindsight Control Plane, implemented on a dedicated branch for isolated testing before merging.

---

## Feature 1: Whitelabel / Brand Style Injector

Enables the control plane UI to be branded per deployment via environment variables.

### Files

| File | Type |
|------|------|
| `src/lib/whitelabel-config.ts` | New — reads `NEXT_PUBLIC_BRAND_*` env vars |
| `src/lib/brand-colors.ts` | New — chart, graph, and gradient constants |
| `src/components/brand-style-injector.tsx` | New — client component, injects CSS custom properties on mount |
| `src/app/layout.tsx` | Modified — dynamic `<title>` and `<meta>` from whitelabel config |
| `src/app/[locale]/layout.tsx` | Modified — mounts `<BrandStyleInjector />` |

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `NEXT_PUBLIC_BRAND_NAME` | `Hindsight` | Brand name throughout UI |
| `NEXT_PUBLIC_BRAND_PRIMARY_COLOR` | `#0074d9` | Primary brand color |
| `NEXT_PUBLIC_BRAND_SECONDARY_COLOR` | `#009296` | Secondary brand color |
| `NEXT_PUBLIC_BRAND_ACCENT_COLOR` | `#f59e0b` | Accent color |
| `NEXT_PUBLIC_BRAND_META_TITLE` | `{name} - Memory...` | Page `<title>` |
| `NEXT_PUBLIC_BRAND_LOGO_URL` | `/logo.png` | Brand logo path |

### How It Works

1. `whitelabel-config.ts` reads `process.env.NEXT_PUBLIC_*` at module init
2. Root layout uses `whitelabelConfig.metadata` for SEO meta tags
3. `BrandStyleInjector` runs on the client to set CSS custom properties (`--brand-primary`, `--primary-gradient`) before first paint

---

## Feature 2: Pending Uploaded Documents Visualizer (#2420)

Monitors file upload operations (`file_convert_retain`) with live progress tracking.

### Files

| File | Type |
|------|------|
| `src/components/pending-uploads-view.tsx` | New — polling component with progress UI |
| `src/components/sidebar.tsx` | Modified — added "Uploads" nav item |
| `src/app/[locale]/banks/[bankId]/page.tsx` | Modified — renders `<PendingUploadsView>` for uploads tab |

### Behavior

- Polls `/api/system/operations` every 5 seconds
- Filters operations by `task_type: file_convert_retain`
- Shows summary cards: Active / Completed / Failed / Total
- Per-operation cards with:
  - Animated progress bar during processing
  - Status badges (Processing/Completed/Failed/Pending)
  - Error messages inline for failed operations
  - Duration tracking and retry count
- Auto-refresh with manual refresh button

### Backend Dependency

Relies on the Hindsight API operations endpoint at `GET /v1/default/banks/{bankId}/operations`. The `/api/system/operations` route proxies this through the control plane.

---

## Feature 3: Search Debug Trace & Score Viewer (#2422)

**No changes required** — `search-debug-view.tsx` (1,031 lines) was already fully implemented.

### Existing Capabilities

- Interactive trace graph showing per-stage retrieval methods
- RRF (Reciprocal Rank Fusion) merge display with per-method score breakdowns
- Cross-encoder reranking scores with tooltips
- Temporal and recency boost indicators
- Combined scoring visualization
- JSON/Results/Trace tab views

---

## Feature 4: S3/Object Storage Config & Prompt Cache Buffer Display

Read-only status cards in the deployment page showing backend configuration.

### Files

| File | Type |
|------|------|
| `src/app/api/system/config/route.ts` | New — API endpoint returning env-derived config values |
| `src/components/deployment-view.tsx` | New — Storage + LLM config cards |
| `src/app/[locale]/deployment/page.tsx` | New — route page |

### API: `GET /api/system/config`

Returns a JSON object with:
```json
{
  "storage_type": "s3 | native",
  "storage_bucket": "my-bucket",
  "storage_region": "us-east-1",
  "llm_provider": "openai | ollama | ...",
  "llm_model": "gpt-4o",
  "prompt_cache_enabled": true,
  "embeddings_provider": "local | openai | ...",
  "reranker_provider": "rrf | cohere | ...",
  "database_type": "pg0 | postgresql"
}
```

### Deployment View Cards

**Storage Config**: Shows storage type (S3 with bucket icon, native with database icon), bucket/region details, and a note about native storage when applicable.

**LLM Config**: Shows LLM provider + model, prompt cache status badge (On/Off with green/gray styling), embeddings model name.

### Backend Env Vars Referenced

| Variable | Purpose |
|----------|---------|
| `HINDSIGHT_API_FILE_STORAGE_TYPE` | `s3` / `native` / `gcs` / `azure` |
| `HINDSIGHT_API_FILE_STORAGE_S3_BUCKET` | S3 bucket name |
| `HINDSIGHT_API_FILE_STORAGE_S3_REGION` | S3 region |
| `HINDSIGHT_API_LLM_PROVIDER` | LLM provider name |
| `HINDSIGHT_API_LLM_MODEL` | Active LLM model |
| `HINDSIGHT_API_LLM_PROMPT_CACHE_ENABLED` | Prompt cache toggle |
| `HINDSIGHT_API_EMBEDDINGS_PROVIDER` | Embeddings provider |
| `HINDSIGHT_API_RERANKER_PROVIDER` | Reranker provider |
| `HINDSIGHT_API_DATABASE_URL` | Database connection |

---

## Feature 5: System Monitor Page (from untracked files)

Comprehensive system monitoring with services grid, operations, and logs.

### Files

| File | Type |
|------|------|
| `src/app/[locale]/system/page.tsx` | New — tabbed system monitor |
| `src/app/api/system/services/route.ts` | New — port-based service detection |
| `src/app/api/system/services/[service]/route.ts` | New — POST start/stop/restart |
| `src/app/api/system/logs/route.ts` | New — log file reader |
| `src/app/api/system/operations/route.ts` | New — proxy to Hindsight API |

### Port Fix Applied

**Control Plane port corrected from 9999 → 9998** in `services/route.ts` (env-configurable via `CONTROL_PLANE_PORT`).

### Services Tab

- Grid of service cards: Hindsight API, Control Plane, Ollama Embeddings, Ollama LLM, PostgreSQL, Workers
- Port check via `lsof -ti:{port}` with PID capture
- Real-time CPU/Memory/uptime via `ps`
- Health check integrations (API health endpoint, Ollama `/api/tags`, PostgreSQL connection status)
- Start/Stop/Restart controls (excludes Control Plane to prevent self-kill)
- Worker scaling (+/-) from 0-10 with state file reading at `/tmp/hindsight-workers.state`

### Operations Tab

- Summary stats: Total / Processing / Pending / Failed
- Operation cards with status badges, age tracking, sticky detection (>5 min)
- Error message display for failed operations
- Auto-refresh with configurable interval

### Logs Tab

- Log viewer with monospace terminal-style display
- Service selector: API, Control Plane, Ollama lanes, Workers
- Reads last 100 lines from log files
- 5-second polling interval

### System API Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/api/system/services` | GET | Check all service ports and health |
| `/api/system/services/[service]` | POST | Start/stop/restart a service |
| `/api/system/logs` | GET | Read service log files |
| `/api/system/operations` | GET | Proxy to Hindsight API operations |
| `/api/system/config` | GET | Return env-derived backend config |

---

## Cross-Cutting: Type Safety

`src/lib/nav-types.ts` was created to share the `NavItem` type between `sidebar.tsx` and the bank page, preventing TypeScript structural type conflicts when both defined the same union type independently.

## Cross-Cutting: i18n

All 10 locale files (`en`, `de`, `fr`, `ja`, `ko`, `pt`, `es`, `zh-CN`, `zh-TW`, `yue-Hant`) updated with:

- `bank.sidebar.uploads` — sidebar nav label
- `uploads` section — 12 keys for the PendingUploadsView component
- `bank.uploads` + `bank.uploadsDescription` — bank page tab
- `deployment` section — keys for the deployment page

---

## Build Verification

```
npm run build  ✓  (TypeScript + standalone pass)
```

Pre-existing warnings only: Next.js 16 `viewport` migration, Python type-checker diagnostics, lockfile issues — all unrelated to these changes.
