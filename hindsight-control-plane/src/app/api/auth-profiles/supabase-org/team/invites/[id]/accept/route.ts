import { NextResponse } from "next/server";

import { acceptInvite, jsonError } from "@/lib/supabase-org/store";

export async function POST(request: Request, { params }: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await params;
    await acceptInvite(request, id);
    return NextResponse.json({ success: true }, { status: 200 });
  } catch (error) {
    return jsonError(error instanceof Error ? error.message : "Failed to accept invite", 400);
  }
}
