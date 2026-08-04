/**
 * Hindsight API clients for the control plane.
 *
 * These are built PER REQUEST, not once at module scope. With tenant auth on,
 * the key used for a downstream call is the one the logged-in person supplied,
 * so a single control plane serves many users and the API — which resolves a
 * key to its own tenant schema — is what enforces the boundary.
 *
 * They stay cheap to construct: no connection pooling, just a base URL and a
 * header. All callers are Next route handlers, so a request scope always exists.
 */

import { cookies } from "next/headers";
import {
  HindsightClient,
  HindsightError,
  createClient,
  createConfig,
  sdk,
} from "@vectorize-io/hindsight-client";

import {
  ACCESS_KEY_COOKIE,
  getSessionSecret,
  isTenantAuthEnabled,
  readTenantSessionKey,
} from "@/lib/auth/session";

export const DATAPLANE_URL = process.env.HINDSIGHT_CP_DATAPLANE_API_URL || "http://localhost:8888";
const DATAPLANE_API_KEY = process.env.HINDSIGHT_CP_DATAPLANE_API_KEY || "";

/**
 * The API key to use for this request.
 *
 * In tenant mode this is the caller's own key, taken from their session, and
 * there is deliberately NO fallback to HINDSIGHT_CP_DATAPLANE_API_KEY: falling
 * back would quietly serve a request with no valid session using the shared
 * key, which is the one failure that would defeat the whole point. Middleware
 * already rejects those requests; this is the second lock.
 */
export async function getDataplaneApiKey(): Promise<string | undefined> {
  if (!isTenantAuthEnabled()) {
    return DATAPLANE_API_KEY || undefined;
  }
  const jar = await cookies();
  const token = jar.get(ACCESS_KEY_COOKIE)?.value;
  const apiKey = await readTenantSessionKey(token, getSessionSecret());
  return apiKey ?? undefined;
}

/**
 * Auth headers for direct fetch calls to the dataplane API.
 */
export async function getDataplaneHeaders(
  extra?: Record<string, string>
): Promise<Record<string, string>> {
  const headers: Record<string, string> = { ...extra };
  const apiKey = await getDataplaneApiKey();
  if (apiKey) {
    headers["Authorization"] = `Bearer ${apiKey}`;
  }
  return headers;
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
export async function getHindsightClient(): Promise<HindsightClient> {
  return new HindsightClient({
    baseUrl: DATAPLANE_URL,
    apiKey: await getDataplaneApiKey(),
  });
}

/**
 * Low-level client for direct SDK access
 */
export async function getLowLevelClient() {
  const apiKey = await getDataplaneApiKey();
  return createClient(
    createConfig({
      baseUrl: DATAPLANE_URL,
      headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : undefined,
    })
  );
}

/**
 * Export SDK functions for direct API access
 */
export { sdk };

/**
 * Export HindsightError for error handling
 */
export { HindsightError };
