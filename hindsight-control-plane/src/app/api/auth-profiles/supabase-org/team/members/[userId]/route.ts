import { NextResponse } from "next/server";

import {
  assertOrganizationRole,
  getCurrentOrgContext,
  jsonError,
  removeMember,
  updateMemberRole,
} from "@/lib/supabase-org/store";
import type { OrganizationRole } from "@/lib/supabase-org/store";

export async function PATCH(request: Request, { params }: { params: Promise<{ userId: string }> }) {
  try {
    const { userId } = await params;
    const body = (await request.json()) as { role?: OrganizationRole };
    if (!body.role) return jsonError("role is required", 400);
    assertOrganizationRole(body.role);
    const context = await getCurrentOrgContext(request);
    const member = await updateMemberRole(context, userId, body.role);
    return NextResponse.json({ member }, { status: 200 });
  } catch (error) {
    return jsonError(error instanceof Error ? error.message : "Failed to update member", 400);
  }
}

export async function DELETE(
  request: Request,
  { params }: { params: Promise<{ userId: string }> }
) {
  try {
    const { userId } = await params;
    const context = await getCurrentOrgContext(request);
    await removeMember(context, userId);
    return NextResponse.json({ success: true }, { status: 200 });
  } catch (error) {
    return jsonError(error instanceof Error ? error.message : "Failed to remove member", 400);
  }
}
