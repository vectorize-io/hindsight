import { NextResponse } from "next/server";

import { getCurrentOrgContext, jsonError, updateOrganizationName } from "@/lib/supabase-org/store";

export async function PATCH(request: Request, { params }: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await params;
    const body = (await request.json()) as { name?: string };
    if (!body.name) return jsonError("name is required", 400);
    const context = await getCurrentOrgContext(request);
    const organization = await updateOrganizationName(context, id, body.name);
    return NextResponse.json({ organization }, { status: 200 });
  } catch (error) {
    return jsonError(error instanceof Error ? error.message : "Failed to update organization", 400);
  }
}
