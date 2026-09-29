import { NextRequest, NextResponse } from "next/server";

import { ACCESS_KEY_COOKIE, sessionCookieOptions } from "@/lib/auth/session";

export async function POST(request: NextRequest) {
  const response = NextResponse.json({ success: true });

  response.cookies.set({
    name: ACCESS_KEY_COOKIE,
    value: "",
    ...sessionCookieOptions(request),
    maxAge: 0,
  });

  return response;
}
