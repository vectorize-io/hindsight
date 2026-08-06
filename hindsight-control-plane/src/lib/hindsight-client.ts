/**
 * Shared Hindsight API client instance for the control plane.
 * Configured to connect to the dataplane API server.
 */

import {
  HindsightClient,
  HindsightError,
  createClient,
  createConfig,
  sdk,
} from "@vectorize-io/hindsight-client";

export const DATAPLANE_URL = process.env.HINDSIGHT_CP_DATAPLANE_API_URL || "http://localhost:8888";
const DATAPLANE_API_KEY = process.env.HINDSIGHT_CP_DATAPLANE_API_KEY || "";

/**
 * Auth headers for direct fetch calls to the dataplane API.
 */
export function getDataplaneHeaders(extra?: Record<string, string>): Record<string, string> {
  const headers: Record<string, string> = { ...extra };
  if (DATAPLANE_API_KEY) {
    headers["Authorization"] = `Bearer ${DATAPLANE_API_KEY}`;
  }
  return headers;
}

/**
 * True when a bank id contains a path separator (`/` or `\`) or a `..` path
 * segment — the shapes that let an id escape its bank path once the URL is
 * normalized. A bank id is a single path segment, so none of these are ever
 * legitimate.
 */
export function bankIdHasTraversal(bankId: string): boolean {
  if (bankId.includes("/") || bankId.includes("\\")) return true;
  // A ".." segment, bounded by start/end or a separator on either side.
  return /(^|[/\\])\.\.([/\\]|$)/.test(bankId);
}

/**
 * Build a dataplane URL for a bank-scoped endpoint with the bank id properly encoded.
 * Bank ids may contain `:`, `%`, etc. (e.g. openclaw `agent::channel::user`),
 * which must be percent-encoded before being interpolated into a URL path.
 *
 * Throws on a traversal-shaped id rather than encoding it. Encoding alone makes
 * the id safe *here*, but a bank id reaching this helper with a `/` or `..` in
 * it is malformed by definition, and callers that build a URL by hand would
 * silently resolve it to a different bank once WHATWG URL normalization
 * collapses the dot segment. Failing loudly keeps that class of bug from being
 * reintroduced by a route that forgets to use this helper.
 */
export function dataplaneBankUrl(bankId: string, suffix = ""): string {
  if (bankIdHasTraversal(bankId)) {
    throw new Error("Invalid bank_id: path separators and '..' segments are not allowed");
  }
  return `${DATAPLANE_URL}/v1/default/banks/${encodeURIComponent(bankId)}${suffix}`;
}

/**
 * High-level client with convenience methods
 */
export const hindsightClient = new HindsightClient({
  baseUrl: DATAPLANE_URL,
  apiKey: DATAPLANE_API_KEY || undefined,
});

/**
 * Low-level client for direct SDK access
 */
export const lowLevelClient = createClient(
  createConfig({
    baseUrl: DATAPLANE_URL,
    headers: DATAPLANE_API_KEY ? { Authorization: `Bearer ${DATAPLANE_API_KEY}` } : undefined,
  })
);

/**
 * Export SDK functions for direct API access
 */
export { sdk };

/**
 * Export HindsightError for error handling
 */
export { HindsightError };
