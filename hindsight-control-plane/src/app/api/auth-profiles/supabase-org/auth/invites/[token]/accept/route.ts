import { NextRequest, NextResponse } from "next/server";

import { SUPABASE_ORG_SELECTED_ORG_COOKIE } from "@/lib/auth-profiles/supabase-org/constants";
import { sessionCookieOptions } from "@/lib/auth/session";
import { acceptInvite, jsonError } from "@/lib/supabase-org/store";

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ token: string }> }
) {
  try {
    const { token } = await params;
    const accepted = await acceptInvite(request, token);
    const response = NextResponse.json(
      { success: true, selected_org_id: accepted.org_id },
      { status: 200 }
    );
    response.cookies.set({
      name: SUPABASE_ORG_SELECTED_ORG_COOKIE,
      value: accepted.org_id,
      ...sessionCookieOptions(request),
      maxAge: 30 * 24 * 60 * 60,
    });
    return response;
  } catch (error) {
    return jsonError(error instanceof Error ? error.message : "Failed to accept invite", 400);
  }
}
