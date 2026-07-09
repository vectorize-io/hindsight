import { NextRequest, NextResponse } from "next/server";

import {
  copySupabaseOrgSessionCookies,
  getCurrentOrgContextForUser,
  getValidSupabaseSession,
  jsonError,
  listOrganizationsForUser,
} from "@/lib/supabase-org/store";

export async function GET(request: NextRequest) {
  const cookieResponse = NextResponse.json({});
  try {
    const { user } = await getValidSupabaseSession(request, cookieResponse);
    const organizations = await listOrganizationsForUser(user.id);
    let current = null;
    try {
      const context = await getCurrentOrgContextForUser(request, user);
      current = {
        org_id: context.selectedOrgId,
        role: context.membership.role,
      };
    } catch {
      current = null;
    }
    const response = NextResponse.json({ user, organizations, current }, { status: 200 });
    copySupabaseOrgSessionCookies(cookieResponse, response);
    return response;
  } catch (error) {
    const response = jsonError(
      error instanceof Error ? error.message : "Failed to resolve current user",
      401
    );
    copySupabaseOrgSessionCookies(cookieResponse, response);
    return response;
  }
}
