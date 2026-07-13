import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import createIntlMiddleware from "next-intl/middleware";

import { ACCESS_KEY_COOKIE, verifySessionToken } from "@/lib/auth/session";
import { bankAllowed } from "@/lib/auth/tokens";
import { stripBasePath, withBasePath } from "@/lib/base-path";
import { routing } from "@/i18n/routing";

// Routes that don't require authentication
const PUBLIC_PATTERNS = [
  "/login",
  "/api/auth/",
  "/api/health",
  "/api/version",
  "/logo.png",
  "/favicon",
  "/_next",
  "/fonts",
  "/static",
];

// API collections whose first path segment after the collection is a bank id.
const BANK_PATH_COLLECTIONS = new Set(["banks", "operations", "stats", "profile"]);

const intlMiddleware = createIntlMiddleware(routing);

/**
 * Resolve the target bank id an authenticated `/api/*` request addresses, from
 * either the path (`/api/<collection>/<id>/...`) or a `?bank_id=`/`?agent_id=`
 * query param. Returns null when no bank id is present (e.g. `/api/banks` list,
 * `/api/version`). Path ids are URL-decoded before comparison.
 */
function apiTargetBankId(appPathname: string, request: NextRequest): string | null {
  const segments = appPathname.split("/").filter(Boolean); // ["api", "<collection>", "<id>", ...]
  if (segments.length >= 3 && segments[0] === "api" && BANK_PATH_COLLECTIONS.has(segments[1])) {
    return decodeURIComponent(segments[2]);
  }

  const query = request.nextUrl.searchParams;
  return query.get("bank_id") ?? query.get("agent_id");
}

function forbidden(request: NextRequest): NextResponse {
  return NextResponse.json(
    localizeApiErrorPayload(request, {
      error: "Forbidden",
      errorKey: "api.errors.auth.forbidden",
    }),
    { status: 403 }
  );
}

/**
 * Origins allowed to embed the Control Plane, read per-request so the runtime
 * env controls framing (Next bakes next.config `headers()` at build time, which
 * would ignore a container-time env). Space- or comma-separated; falls back to
 * `'self'` when unset, never `*`.
 */
function frameAncestorsValue(): string {
  return (process.env.HINDSIGHT_CP_FRAME_ANCESTORS || "'self'")
    .split(/[\s,]+/)
    .filter(Boolean)
    .join(" ");
}

function withFrameAncestors(response: NextResponse): NextResponse {
  response.headers.set("Content-Security-Policy", `frame-ancestors ${frameAncestorsValue()};`);
  return response;
}

export async function middleware(request: NextRequest): Promise<NextResponse> {
  return withFrameAncestors(await handle(request));
}

async function handle(request: NextRequest): Promise<NextResponse> {
  const accessKey = process.env.HINDSIGHT_CP_ACCESS_KEY;
  const { pathname } = request.nextUrl;
  const appPathname = stripBasePath(pathname);

  // API routes are not locale-prefixed — handle auth directly without i18n routing.
  if (appPathname.startsWith("/api/")) {
    if (!accessKey) {
      return NextResponse.next();
    }

    const isPublic = PUBLIC_PATTERNS.some((pattern) => appPathname.startsWith(pattern));
    if (isPublic) {
      return NextResponse.next();
    }

    const sessionCookie = request.cookies.get(ACCESS_KEY_COOKIE)?.value;
    const session = await verifySessionToken(sessionCookie, accessKey);

    if (!session.valid) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "Unauthorized",
          errorKey: "api.errors.auth.unauthorized",
        }),
        { status: 401 }
      );
    }

    // Scoped sessions may only reach banks under their prefix. Body-only bank
    // ids are enforced in-route via bank-guard (middleware can't read the body).
    if (session.prefix) {
      const targetBankId = apiTargetBankId(appPathname, request);
      if (targetBankId !== null && !bankAllowed(session.prefix, targetBankId)) {
        return forbidden(request);
      }
    }

    return NextResponse.next();
  }

  // Page routes: enforce auth first, then delegate to the i18n middleware for
  // locale negotiation and rewriting. With localePrefix "never" the locale is
  // never in the path, so appPathname is already the canonical route.
  if (accessKey) {
    const isPublic = PUBLIC_PATTERNS.some((pattern) => appPathname.startsWith(pattern));

    if (!isPublic) {
      const sessionCookie = request.cookies.get(ACCESS_KEY_COOKIE)?.value;
      const session = await verifySessionToken(sessionCookie, accessKey);

      if (!session.valid) {
        // Next.js middleware redirects do not automatically inherit next.config basePath.
        // Prefix the target explicitly, but keep returnTo as the app-relative path so
        // client-side router.push() does not double-prefix after login.
        const loginUrl = new URL(withBasePath("/login"), request.url);
        loginUrl.searchParams.set("returnTo", appPathname);
        return NextResponse.redirect(loginUrl);
      }

      // Block scoped sessions from navigating to a foreign bank page; send them
      // back to the dashboard rather than exposing another prefix's chrome.
      if (session.prefix) {
        const bankMatch = appPathname.match(/^\/banks\/([^/?]+)/);
        if (bankMatch && !bankAllowed(session.prefix, decodeURIComponent(bankMatch[1]))) {
          return NextResponse.redirect(new URL(withBasePath("/dashboard"), request.url));
        }
      }
    }
  }

  return intlMiddleware(request);
}

export const config = {
  // Match all paths except Next.js internals and static assets.
  // - Use an explicit file extension allowlist instead of .*\..* so that
  //   dynamic segments containing dots (e.g. bank IDs like
  //   "SX.Products.GovComply.Build") still get the i18n locale rewrite.
  matcher:
    "/((?!_next|_vercel|.*\\.(?:png|jpe?g|gif|svg|webp|ico|css|js|map|woff2?|ttf|eot|txt|xml|json)$).*)",
};
