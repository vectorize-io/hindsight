import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

import { ACCESS_KEY_COOKIE, verifySessionToken } from "@/lib/auth/session";

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

export async function middleware(request: NextRequest) {
  const accessKey = process.env.HINDSIGHT_CP_ACCESS_KEY;

  // If no access key is configured, skip auth entirely
  if (!accessKey) {
    return NextResponse.next();
  }

  const { pathname } = request.nextUrl;

  // Check if this path is public
  const isPublic = PUBLIC_PATTERNS.some((pattern) => pathname.startsWith(pattern));

  if (isPublic) {
    return NextResponse.next();
  }

  const sessionCookie = request.cookies.get(ACCESS_KEY_COOKIE)?.value;
  const isAuthenticated = await verifySessionToken(sessionCookie, accessKey);

  if (!isAuthenticated) {
    // For API routes, return 401 JSON instead of redirecting to HTML login page
    if (pathname.startsWith("/api/")) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    // Redirect to login page
    const loginUrl = new URL("/login", request.url);
    loginUrl.searchParams.set("returnTo", pathname);
    return NextResponse.redirect(loginUrl);
  }

  return NextResponse.next();
}

export const config = {
  // Match all routes
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
