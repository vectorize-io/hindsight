/**
 * Helpers for building URLs that include a bank id.
 *
 * Bank ids are user-defined and may contain characters that are not URL-safe
 * (e.g. openclaw composite ids like `agent-1::channel-2::user-3`, which contain
 * `:` and may also contain `/`, `%`, spaces, etc.). They must be percent-encoded
 * before being interpolated into a URL path or query string — both for client
 * navigation (`/banks/...`) and for calls to the control-plane proxy
 * (`/api/banks/...`).
 *
 * Always use these helpers instead of raw template literals.
 */

const enc = (value: string): string => encodeURIComponent(value);

/** Page route for a bank in the control plane app router. */
export function bankRoute(bankId: string, suffix = ""): string {
  return `/banks/${enc(bankId)}${suffix}`;
}

/** Control-plane proxy URL under `/api/banks/...` for a bank-scoped endpoint. */
export function bankApi(bankId: string, suffix = ""): string {
  return `/api/banks/${enc(bankId)}${suffix}`;
}

/** Control-plane proxy URL under `/api/stats/...` for bank statistics. */
export function bankStatsApi(bankId: string, suffix = ""): string {
  return `/api/stats/${enc(bankId)}${suffix}`;
}

/** Control-plane proxy URL for memory operations scoped to a bank via query string. */
export function memoryApi(memoryId: string, bankId: string, suffix = ""): string {
  return `/api/memories/${enc(memoryId)}${suffix}${suffix.includes("?") ? "&" : "?"}bank_id=${enc(bankId)}`;
}

/**
 * Control-plane proxy URL for document operations.
 *
 * Document ids are user-supplied and routinely contain `/` (S3 keys, file paths).
 * They stay in the query string, never in a path segment: ingress proxies decode
 * `%2F` back to `/` during path normalization (Azure Container Apps, AWS ALB), which
 * splits the id across segments and 404s the route. Query strings are left alone.
 */
export function documentApi(documentId: string, bankId: string, suffix = ""): string {
  return `/api/documents${suffix}?bank_id=${enc(bankId)}&document_id=${enc(documentId)}`;
}

/** Control-plane proxy URL for a single chunk. See `documentApi` on why the id is a query param. */
export function chunkApi(chunkId: string): string {
  return `/api/chunks?chunk_id=${enc(chunkId)}`;
}
