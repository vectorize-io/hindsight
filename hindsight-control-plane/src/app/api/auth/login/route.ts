import { NextRequest, NextResponse } from "next/server";

const ACCESS_KEY_COOKIE = "hindsight_cp_access";

export async function POST(request: NextRequest) {
  const accessKey = process.env.HINDSIGHT_CP_ACCESS_KEY;

  // If no access key is configured, return 503
  if (!accessKey) {
    return NextResponse.json({ error: "Access key not configured" }, { status: 503 });
  }

  let body: { key?: string };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const providedKey = body.key;

  // Constant-time comparison to prevent timing attacks
  const isValid = providedKey && constantTimeCompare(providedKey, accessKey);

  if (!isValid) {
    return NextResponse.json({ error: "Invalid access key" }, { status: 401 });
  }

  const response = NextResponse.json({ success: true });

  // Set HttpOnly, Secure, SameSite cookie
  response.cookies.set({
    name: ACCESS_KEY_COOKIE,
    value: "authenticated",
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: 60 * 60 * 24, // 24 hours
  });

  return response;
}

/**
 * Constant-time string comparison to prevent timing attacks.
 */
function constantTimeCompare(a: string, b: string): boolean {
  if (a.length !== b.length) {
    return false;
  }

  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }

  return result === 0;
}
