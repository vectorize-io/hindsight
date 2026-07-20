import { NextResponse } from "next/server";

import {
  assertOrganizationRole,
  createInvite,
  getCurrentOrgContext,
  jsonError,
  listInvites,
} from "@/lib/supabase-org/store";
import type { OrganizationRole } from "@/lib/supabase-org/store";

export async function GET(request: Request) {
  try {
    const context = await getCurrentOrgContext(request);
    return NextResponse.json(
      { invites: await listInvites(context.selectedOrgId) },
      { status: 200 }
    );
  } catch (error) {
    return jsonError(error instanceof Error ? error.message : "Failed to list invites", 400);
  }
}

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as { email?: string; role?: OrganizationRole };
    if (!body.email) return jsonError("email is required", 400);
    if (body.role) assertOrganizationRole(body.role);
    const context = await getCurrentOrgContext(request);
    const invite = await createInvite(context, body.email, body.role || "member");
    return NextResponse.json({ invite }, { status: 201 });
  } catch (error) {
    return jsonError(error instanceof Error ? error.message : "Failed to create invite", 400);
  }
}
