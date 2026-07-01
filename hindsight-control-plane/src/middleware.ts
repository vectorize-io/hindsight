import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import createIntlMiddleware from "next-intl/middleware";

import { ACCESS_KEY_COOKIE, verifySessionToken } from "@/lib/auth/session";
import {
  SUPABASE_ORG_ACCESS_TOKEN_COOKIE,
  SUPABASE_ORG_SELECTED_ORG_COOKIE,
  getControlPlaneAuthProvider,
  isSupabaseOrgSessionPresent,
} from "@/lib/auth/provider";
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

const intlMiddleware = createIntlMiddleware(routing);

export async function middleware(request: NextRequest) {
  const authProvider = getControlPlaneAuthProvider();
  const { pathname } = request.nextUrl;
  const appPathname = stripBasePath(pathname);

  // API routes are not locale-prefixed — handle auth directly without i18n routing.
  if (appPathname.startsWith("/api/")) {
    if (authProvider === "disabled") {
      return NextResponse.next();
    }

    const isPublic = PUBLIC_PATTERNS.some((pattern) => appPathname.startsWith(pattern));
    if (isPublic) {
      return NextResponse.next();
    }

    if (!(await isRequestAuthenticated(request, authProvider))) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "Unauthorized",
          errorKey: "api.errors.auth.unauthorized",
        }),
        { status: 401 }
      );
    }

    return NextResponse.next();
  }

  // Page routes: enforce auth first, then delegate to the i18n middleware for
  // locale negotiation and rewriting. With localePrefix "never" the locale is
  // never in the path, so appPathname is already the canonical route.
  if (authProvider !== "disabled") {
    const isPublic = PUBLIC_PATTERNS.some((pattern) => appPathname.startsWith(pattern));

    if (!isPublic) {
      if (!(await isRequestAuthenticated(request, authProvider))) {
        // Next.js middleware redirects do not automatically inherit next.config basePath.
        // Prefix the target explicitly, but keep returnTo as the app-relative path so
        // client-side router.push() does not double-prefix after login.
        const loginUrl = new URL(withBasePath("/login"), request.url);
        loginUrl.searchParams.set("returnTo", appPathname);
        return NextResponse.redirect(loginUrl);
      }
    }
  }

  return intlMiddleware(request);
}

async function isRequestAuthenticated(
  request: NextRequest,
  authProvider: ReturnType<typeof getControlPlaneAuthProvider>
): Promise<boolean> {
  if (authProvider === "disabled") {
    return true;
  }
  if (authProvider === "access_key") {
    return verifySessionToken(
      request.cookies.get(ACCESS_KEY_COOKIE)?.value,
      process.env.HINDSIGHT_CP_ACCESS_KEY || ""
    );
  }
  return isSupabaseOrgSessionPresent(
    request.cookies.get(SUPABASE_ORG_ACCESS_TOKEN_COOKIE)?.value,
    request.cookies.get(SUPABASE_ORG_SELECTED_ORG_COOKIE)?.value
  );
}

export const config = {
  // Match all paths except Next.js internals and static assets.
  // - Use an explicit file extension allowlist instead of .*\..* so that
  //   dynamic segments containing dots (e.g. bank IDs like
  //   "SX.Products.GovComply.Build") still get the i18n locale rewrite.
  matcher:
    "/((?!_next|_vercel|.*\\.(?:png|jpe?g|gif|svg|webp|ico|css|js|map|woff2?|ttf|eot|txt|xml|json)$).*)",
};
