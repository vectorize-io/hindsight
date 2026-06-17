import { NextResponse } from "next/server";

import {
  getAuthenticatedUser,
  getCurrentOrgContext,
  jsonError,
  listOrganizationsForUser,
} from "@/lib/supabase-org/store";

export async function GET(request: Request) {
  try {
    const user = await getAuthenticatedUser(request);
    const organizations = await listOrganizationsForUser(user.id);
    let current = null;
    try {
      const context = await getCurrentOrgContext(request);
      current = {
        org_id: context.selectedOrgId,
        role: context.membership.role,
      };
    } catch {
      current = null;
    }
    return NextResponse.json({ user, organizations, current }, { status: 200 });
  } catch (error) {
    return jsonError(
      error instanceof Error ? error.message : "Failed to resolve current user",
      401
    );
  }
}
