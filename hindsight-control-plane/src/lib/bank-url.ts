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

/** Control-plane proxy URL for document operations scoped to a bank via query string. */
export function documentApi(documentId: string, bankId: string): string {
  return `/api/documents/${enc(documentId)}?bank_id=${enc(bankId)}`;
}
