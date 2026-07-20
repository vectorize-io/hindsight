import { NextRequest, NextResponse } from "next/server";

import {
  clearSupabaseOrgSessionCookies,
  copySupabaseOrgSessionCookies,
  getValidSupabaseSession,
  jsonError,
  signInWithPassword,
  signOutSupabaseSession,
  updateSupabaseUserPassword,
} from "@/lib/supabase-org/store";

export async function POST(request: NextRequest) {
  const response = NextResponse.json({ success: true }, { status: 200 });
  try {
    const body = (await request.json()) as {
      current_password?: string;
      new_password?: string;
    };
    if (!body.current_password || !body.new_password) {
      return jsonError("current_password and new_password are required", 400);
    }

    const session = await getValidSupabaseSession(request, response);
    const reauthSession = await signInWithPassword(session.user.email, body.current_password);
    try {
      await updateSupabaseUserPassword(session.accessToken, body.new_password);
    } catch (error) {
      await bestEffortSignOut(reauthSession.access_token, "local");
      throw error;
    }
    await bestEffortSignOut(session.accessToken, "global");
    await bestEffortSignOut(reauthSession.access_token, "local");
    clearSupabaseOrgSessionCookies(response, request);
    return response;
  } catch (error) {
    const errorResponse = jsonError(
      error instanceof Error ? error.message : "Failed to change password",
      400
    );
    copySupabaseOrgSessionCookies(response, errorResponse);
    return errorResponse;
  }
}

async function bestEffortSignOut(
  accessToken: string | undefined,
  scope: "local" | "global" | "others"
): Promise<void> {
  try {
    await signOutSupabaseSession(accessToken, scope);
  } catch {
    // Password-change cleanup should not leave the browser logged in just because Supabase logout failed.
  }
}
