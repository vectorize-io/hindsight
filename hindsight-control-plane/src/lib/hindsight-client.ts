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

import {
  dataplaneHeadersFor,
  resolveDataplaneAuthHeader,
} from "@/lib/auth/dataplane-auth";

export const DATAPLANE_URL = process.env.HINDSIGHT_CP_DATAPLANE_API_URL || "http://localhost:8888";
const DATAPLANE_API_KEY = process.env.HINDSIGHT_CP_DATAPLANE_API_KEY || "";

/**
 * Minimal shape of an inbound request needed to resolve auth — anything with a
 * header getter (Next.js `Request` / `NextRequest` both satisfy this).
 */
type InboundRequest = { headers: Headers | { get(name: string): string | null } };

/**
 * Auth headers for direct fetch calls to the dataplane API.
 *
 * When an inbound `request` is supplied, auth is resolved per-request via the
 * cascade in {@link dataplaneHeadersFor} (forwarded user token > static key >
 * none) — this is what route handlers acting on behalf of a user should pass.
 *
 * When `request` is omitted (or `null`), it falls back to the static key only —
 * for server-side contexts with no end user (health checks, background tasks).
 */
export function getDataplaneHeaders(
  request?: InboundRequest | null,
  extra?: Record<string, string>
): Record<string, string> {
  if (request) {
    return dataplaneHeadersFor(request, extra);
  }
  const headers: Record<string, string> = { ...extra };
  if (DATAPLANE_API_KEY) {
    headers["Authorization"] = `Bearer ${DATAPLANE_API_KEY}`;
  }
  return headers;
}

/**
 * Build a request-scoped low-level SDK client whose `Authorization` header is
 * resolved from the inbound request (forwarded user token > static key > none).
 *
 * Route handlers that act on behalf of an end user should use this instead of
 * the shared {@link lowLevelClient} singleton, so the user's identity is
 * carried through to the dataplane.
 */
export function getDataplaneClient(request: InboundRequest) {
  const auth = resolveDataplaneAuthHeader(request);
  return createClient(
    createConfig({
      baseUrl: DATAPLANE_URL,
      headers: auth ? { Authorization: auth } : undefined,
    })
  );
}

/**
 * Build a request-scoped high-level {@link HindsightClient} whose auth is
 * resolved from the inbound request (forwarded user token > static key > none).
 *
 * Use this instead of the shared {@link hindsightClient} singleton in route
 * handlers that act on behalf of an end user.
 */
export function getDataplaneHindsightClient(request: InboundRequest): HindsightClient {
  const auth = resolveDataplaneAuthHeader(request);
  // HindsightClient takes an apiKey and prefixes it with "Bearer ". Strip a
  // leading "Bearer " from the resolved header so we don't double-prefix.
  const apiKey = auth?.replace(/^Bearer\s+/i, "") || undefined;
  return new HindsightClient({
    baseUrl: DATAPLANE_URL,
    apiKey,
  });
}

/**
 * Build a dataplane URL for a bank-scoped endpoint with the bank id properly encoded.
 * Bank ids may contain `:`, `/`, `%`, etc. (e.g. openclaw `agent::channel::user`),
 * which must be percent-encoded before being interpolated into a URL path.
 */
export function dataplaneBankUrl(bankId: string, suffix = ""): string {
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
