# CollabMind Operator GUI v0.1

A minimal Next.js (App Router) operator console. It talks **only** to the CollabMind
Central API — never to Google Drive, Qdrant, or the database directly. That boundary is the
whole point of the control plane.

## Pages
Login · Workspaces · Connectors · Google Drive Connection · Indexed Files · Ingestion Queue ·
Audit Log · Agent Activity · Permissions View.

## Operator actions
Connect / disconnect Google Drive · show connector status · trigger manual sync · list discovered
files · show ingestion jobs and process them · inspect permission snapshots · view audit events ·
view agent/MCP activity (and simulate a governed agent request) · disable a source document.

## Run
```bash
cp .env.local.example .env.local   # set NEXT_PUBLIC_CENTRAL_API_URL (default http://localhost:8000)
npm install
npm run dev                        # http://localhost:3001
```
Start the Central API first (see ../README — `uvicorn app.main:app --reload`).

## Boundaries (enforced by design)
- Single network surface: `lib/api.ts`. There is no Google/Qdrant/DB client in this app.
- No secrets or tokens are stored in the GUI; auth is the Central API's concern (dev identity in
  v0.1; Authentik/OIDC later).
- Retrieval/permissions are governed server-side and fail closed; the GUI only displays state.
