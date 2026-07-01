# API-UI-CONTRACT-001

## Scope

This document defines the safe typed API client contract for the Operator Panel base UI. The implementation lives in `operator-ui/lib/operatorApiClient.ts`.

This contract is API client only. It does not add UI screens, fake data, backend route behavior, stored credentials, or localhost-only assumptions.

## Existing Operator UI Source

- Operator UI root: `operator-ui/`
- Existing UI API helper: `operator-ui/lib/api.ts`
- Typed contract client: `operator-ui/lib/operatorApiClient.ts`
- API contract tests: `operator-ui/lib/operatorApiClient.test.ts`
- Test command: `npm run test:api-contract`
- Config example: `operator-ui/.env.local.example`
- Current UI env key: `NEXT_PUBLIC_CENTRAL_API_URL`

The existing `operator-ui/lib/api.ts` is still the v0.1 UI helper used by current pages. UI-002 should migrate new connection-aware screens to `operatorApiClient.ts` rather than expanding the old helper.

## Connection Config

The client accepts this in-memory model:

```ts
type ConnectionConfig = {
  apiBaseUrl: string;
  apiKey?: string;
  jwtToken?: string;
  mode: "local" | "remote";
  connected: boolean;
  lastCheckedAt?: string;
  lastError?: string;
};
```

UI-002 should collect `apiBaseUrl`, `apiKey`, and optional `jwtToken` from operator-entered GUI fields. Do not write tokens or API keys into the repo, `.env.local.example`, screenshots, logs, or test fixtures.

## Auth Behavior

The request wrapper normalizes the base URL and injects auth headers on every request.

- JWT mode: when `jwtToken` is present, send `Authorization: Bearer <jwtToken>`.
- API key mode: when only `apiKey` is present, send both `Authorization: Bearer <apiKey>` and `X-Api-Key: <apiKey>`.
- JWT takes precedence over API key if both are supplied.
- Errors, details, and logs are passed through secret masking before they leave the client.

## Result Shape

Every client method returns one normalized result shape:

```ts
type ApiResult<T> =
  | { ok: true; data: T; status: number }
  | { ok: false; error: ApiError; status: number };
```

Do not fake success in UI code. Render `ok: false` states as connection or route errors.

## Routes Used

| Client method | Method | Route |
| --- | --- | --- |
| `health()` | GET | `/health`, fallback to `/api/health`, then `/api/health/engines` |
| `validateAuth()` | GET | `/me` |
| `memorySearch()` | POST | `/api/retrieval/search?query=...&mode=...&limit=...` |
| `auditEvents()` | GET | `/api/audit/events?limit=...` |
| `governanceQuarantine()` | GET | `/api/gov/approval-queue?status=...&limit=...` |
| `governancePolicyCheck()` | POST | `/api/gov/policy-check` |
| `executionLedger()` | GET | `/api/executions/history?status=...&limit=...&offset=...` |
| `adapterStatus()` | GET | `/api/health/engines` |
| `serviceHealth()` | GET | `/api/observability/service-status` |

## Health Fallback

Health is the only method with route fallback. It tries:

1. `/health`
2. `/api/health`
3. `/api/health/engines`

Fallback only continues for `404` or `405`. Any other failure returns the normalized failure from the last attempted route. The client does not fake success.

## Unsupported Endpoints / Contract Blockers

Current backend route declarations do not include `POST /api/gov/policy-check`. The typed client exposes `governancePolicyCheck()` because the Operator Panel needs this contract, but the current backend will return a clean `ok: false` response until the route exists.

No backend behavior was changed for this contract task.

## UI-002 Consumption

UI-002 should:

1. Store connection fields only in client-side runtime state or approved browser storage.
2. Create a client with `createOperatorApiClient(connectionConfig)`.
3. Call `health()` after the operator enters a base URL and credential.
4. Set `connected`, `lastCheckedAt`, and `lastError` from the returned `ApiResult`.
5. Use the typed methods directly for panel data and show `ok: false` errors without inventing placeholder data.
6. Keep Playground, Chat, Voice, and advanced dashboard calls out of this client until those contracts are approved.

## Validation

Run from `operator-ui/`:

```bash
npm run test:api-contract
npm run typecheck
```

Run `npm run build` or existing console tests if the change touches UI rendering. This task only touches the API contract client and documentation.
