import { NextResponse } from "next/server";

import { getCurrentOrgContext, jsonError, revokeInvite } from "@/lib/supabase-org/store";

export async function DELETE(request: Request, { params }: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await params;
    const context = await getCurrentOrgContext(request);
    await revokeInvite(context, id);
    return NextResponse.json({ success: true }, { status: 200 });
  } catch (error) {
    return jsonError(error instanceof Error ? error.message : "Failed to revoke invite", 400);
  }
}
