import { NextRequest, NextResponse } from "next/server";

import { SUPABASE_ORG_SELECTED_ORG_COOKIE } from "@/lib/auth-profiles/supabase-org/constants";
import { sessionCookieOptions } from "@/lib/auth/session";
import {
  copySupabaseOrgSessionCookies,
  getAuthenticatedUserWithRefresh,
  jsonError,
  listOrganizationsForUser,
} from "@/lib/supabase-org/store";

export async function POST(request: NextRequest) {
  const response = NextResponse.json({}, { status: 200 });
  try {
    const body = (await request.json()) as { org_id?: string };
    if (!body.org_id) return jsonError("org_id is required", 400);
    const user = await getAuthenticatedUserWithRefresh(request, response);
    const organizations = await listOrganizationsForUser(user.id);
    if (!organizations.some((organization) => organization.id === body.org_id)) {
      const errorResponse = jsonError("User is not a member of the selected organization", 403);
      copySupabaseOrgSessionCookies(response, errorResponse);
      return errorResponse;
    }
    response.cookies.set({
      name: SUPABASE_ORG_SELECTED_ORG_COOKIE,
      value: body.org_id,
      ...sessionCookieOptions(request),
      maxAge: 30 * 24 * 60 * 60,
    });
    const successResponse = NextResponse.json(
      { success: true, selected_org_id: body.org_id },
      { status: 200 }
    );
    copySupabaseOrgSessionCookies(response, successResponse);
    return successResponse;
  } catch (error) {
    const errorResponse = jsonError(
      error instanceof Error ? error.message : "Failed to select organization",
      400
    );
    copySupabaseOrgSessionCookies(response, errorResponse);
    return errorResponse;
  }
}
