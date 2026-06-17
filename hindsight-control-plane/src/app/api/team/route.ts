import { NextResponse } from "next/server";

import { getCurrentOrgContext, jsonError, listMembers } from "@/lib/supabase-org/store";

export async function GET(request: Request) {
  try {
    const context = await getCurrentOrgContext(request);
    return NextResponse.json(
      { members: await listMembers(context.selectedOrgId) },
      { status: 200 }
    );
  } catch (error) {
    return jsonError(error instanceof Error ? error.message : "Failed to list team", 400);
  }
}
