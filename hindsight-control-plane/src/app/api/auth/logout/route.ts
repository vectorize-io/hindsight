import { NextRequest, NextResponse } from "next/server";

import { ACCESS_KEY_COOKIE, sessionCookieOptions } from "@/lib/auth/session";
import { clearSupabaseOrgSessionCookies } from "@/lib/supabase-org/store";

export async function POST(request: NextRequest) {
  const response = NextResponse.json({ success: true });

  response.cookies.set({
    name: ACCESS_KEY_COOKIE,
    value: "",
    ...sessionCookieOptions(request),
    maxAge: 0,
  });
  clearSupabaseOrgSessionCookies(response, request);

  return response;
}
