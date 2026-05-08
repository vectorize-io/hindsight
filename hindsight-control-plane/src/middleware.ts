import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

const ACCESS_KEY_COOKIE = "hindsight_cp_access";

// Routes that don't require authentication
const PUBLIC_PATTERNS = [
  "/login",
  "/api/auth/",
  "/api/health",
  "/api/version",
  "/favicon",
  "/_next",
  "/fonts",
  "/static",
];

export function middleware(request: NextRequest) {
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

  // Check for the session cookie
  const isAuthenticated = request.cookies.has(ACCESS_KEY_COOKIE);

  if (!isAuthenticated) {
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
