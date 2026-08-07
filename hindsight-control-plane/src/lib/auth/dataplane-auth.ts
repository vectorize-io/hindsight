/**
 * Request-scoped dataplane authentication resolution.
 *
 * The Control Plane is a backend-for-frontend (BFF): browser requests hit
 * Next.js route handlers, which in turn call the dataplane API. How those
 * downstream calls authenticate depends on how the deployment is fronted.
 *
 * Auth-mode cascade (highest priority first):
 *
 *   1. Forwarded `Authorization` header — when the Control Plane runs behind a
 *      reverse proxy / identity-aware proxy that authenticates the end user and
 *      forwards their bearer token (e.g. oauth2-proxy with
 *      `--pass-access-token`, an API gateway, or a direct API caller), we
 *      forward that exact token to the dataplane. This preserves per-user
 *      identity end-to-end, so the dataplane can enforce per-user tenant
 *      isolation. The Control Plane stays auth-agnostic: it trusts whatever
 *      `Authorization` the upstream proxy injected.
 *
 *   2. Static dataplane API key — `HINDSIGHT_CP_DATAPLANE_API_KEY`. A single
 *      shared credential for all Control Plane traffic. This is the historical
 *      behavior for self-hosted single-tenant deployments.
 *
 *   3. No auth — the dataplane is reachable without credentials (local dev or
 *      an internally-trusted network).
 *
 * Keeping this generic (we forward an opaque `Authorization` header rather than
 * hard-coding any specific provider) means the same code path works for GitHub
 * OAuth, OIDC, API gateways, or static keys, and is safe to merge upstream
 * without coupling the Control Plane to any particular IdP.
 */

const STATIC_DATAPLANE_API_KEY = process.env.HINDSIGHT_CP_DATAPLANE_API_KEY || "";

/**
 * Additional inbound headers that identity-aware proxies use to carry the
 * end-user's access token. Checked only when a raw `Authorization` header is
 * not already present. Order is significant (first match wins).
 */
const FORWARDED_TOKEN_HEADERS = [
  "x-forwarded-access-token",
  "x-auth-request-access-token",
] as const;

/**
 * Resolve the `Authorization` header value the dataplane call should use for a
 * given inbound request, following the cascade documented above.
 *
 * @returns the full header value (e.g. `"Bearer abc123"`) or `undefined` when
 *          no credential applies (open dataplane).
 */
export function resolveDataplaneAuthHeader(
  request: { headers: Headers | { get(name: string): string | null } }
): string | undefined {
  const get = (name: string) => request.headers.get(name);

  // 1. Pass through an Authorization header injected upstream / by the caller.
  const inboundAuth = get("authorization");
  if (inboundAuth && inboundAuth.trim().length > 0) {
    return inboundAuth;
  }

  // 1b. Some proxies forward only the raw token in a side header.
  for (const headerName of FORWARDED_TOKEN_HEADERS) {
    const token = get(headerName);
    if (token && token.trim().length > 0) {
      return `Bearer ${token.trim()}`;
    }
  }

  // 2. Fall back to the static shared dataplane key, if configured.
  if (STATIC_DATAPLANE_API_KEY) {
    return `Bearer ${STATIC_DATAPLANE_API_KEY}`;
  }

  // 3. No credential — open dataplane.
  return undefined;
}

/**
 * Build a header object for direct `fetch` calls to the dataplane, merging the
 * resolved `Authorization` header with any caller-provided extras.
 */
export function dataplaneHeadersFor(
  request: { headers: Headers | { get(name: string): string | null } },
  extra?: Record<string, string>
): Record<string, string> {
  const headers: Record<string, string> = { ...extra };
  const auth = resolveDataplaneAuthHeader(request);
  if (auth) {
    headers["Authorization"] = auth;
  }
  return headers;
}

/**
 * True when this deployment is fronted by an identity-aware proxy that has
 * already authenticated the end user (i.e. a usable forwarded credential is
 * present on the request). Used by the UI auth gate to decide whether the
 * Control Plane's own login is required.
 */
export function hasForwardedIdentity(
  request: { headers: Headers | { get(name: string): string | null } }
): boolean {
  const get = (name: string) => request.headers.get(name);
  if ((get("authorization") || "").trim().length > 0) return true;
  return FORWARDED_TOKEN_HEADERS.some((h) => (get(h) || "").trim().length > 0);
}
