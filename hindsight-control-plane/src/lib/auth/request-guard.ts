import type { NextRequest } from "next/server";

// API collections whose first path segment after the collection is a bank id.
const BANK_PATH_COLLECTIONS = new Set(["banks", "operations", "stats", "profile"]);

/**
 * Resolve the target bank id an authenticated `/api/*` request addresses, from
 * either the path (`/api/<collection>/<id>/...`) or a `?bank_id=`/`?agent_id=`
 * query param. Returns null when no bank id is present (e.g. `/api/banks` list,
 * `/api/version`). Path ids are URL-decoded before comparison.
 */
export function apiTargetBankId(appPathname: string, request: NextRequest): string | null {
  const segments = appPathname.split("/").filter(Boolean); // ["api", "<collection>", "<id>", ...]
  if (segments.length >= 3 && segments[0] === "api" && BANK_PATH_COLLECTIONS.has(segments[1])) {
    return decodeURIComponent(segments[2]);
  }

  const query = request.nextUrl.searchParams;
  return query.get("bank_id") ?? query.get("agent_id");
}

/** Public host the client actually addressed, honoring the nginx proxy. */
function requestHost(request: NextRequest): string {
  return (
    request.headers.get("x-forwarded-host")?.split(",")[0]?.trim() ||
    request.headers.get("host")?.trim() ||
    request.nextUrl.host
  );
}

/**
 * Absolute origins permitted to make cross-site state-changing calls, derived
 * from `HINDSIGHT_CP_FRAME_ANCESTORS` (the same allowlist that authorizes
 * framing). `'self'`/`'none'` are CSP keywords, not origins, so they're
 * dropped; anything that doesn't parse as a URL is ignored.
 */
function allowedEmbedOrigins(): string[] {
  return (process.env.HINDSIGHT_CP_FRAME_ANCESTORS || "")
    .split(/[\s,]+/)
    .filter((value) => value && value !== "'self'" && value !== "'none'")
    .flatMap((value) => {
      try {
        return [new URL(value).origin];
      } catch {
        return [];
      }
    });
}

/**
 * CSRF guard for state-changing `/api/*` calls. With `SameSite=None` the session
 * cookie rides along on cross-site requests, so `SameSite` no longer blocks
 * CSRF; this restores that defense via an `Origin` check. A non-GET request is
 * cross-site when its `Origin` is neither same-origin nor an allowed embed
 * origin. When no `Origin` is present we fall back to `Sec-Fetch-Site`; when
 * neither header is present (non-browser API clients) we allow, since a browser
 * CSRF always carries one of them.
 */
export function isCrossSiteWrite(request: NextRequest): boolean {
  const method = request.method.toUpperCase();
  if (method === "GET" || method === "HEAD" || method === "OPTIONS") return false;

  const origin = request.headers.get("origin");
  if (origin) {
    let originUrl: URL;
    try {
      originUrl = new URL(origin);
    } catch {
      return true;
    }
    if (originUrl.host === requestHost(request)) return false;
    return !allowedEmbedOrigins().includes(originUrl.origin);
  }

  const secFetchSite = request.headers.get("sec-fetch-site");
  if (secFetchSite) {
    return secFetchSite !== "same-origin" && secFetchSite !== "none";
  }

  return false;
}
