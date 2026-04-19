import { NextRequest, NextResponse } from "next/server";
import { hindsightClient, sdk, lowLevelClient } from "@/lib/hindsight-client";

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const bankId = searchParams.get("bank_id") || searchParams.get("agent_id");

    if (!bankId) {
      return NextResponse.json({ error: "bank_id is required" }, { status: 400 });
    }

    const limit = searchParams.get("limit") ? Number(searchParams.get("limit")) : undefined;
    const offset = searchParams.get("offset") ? Number(searchParams.get("offset")) : undefined;
    const type = searchParams.get("type") || searchParams.get("fact_type") || undefined;
    const q = searchParams.get("q") || undefined;
    const consolidationStateParam =
      searchParams.get("consolidation_state") || searchParams.get("consolidationState");
    const consolidationState =
      consolidationStateParam === "failed" ||
      consolidationStateParam === "pending" ||
      consolidationStateParam === "done"
        ? consolidationStateParam
        : undefined;

    const response = await hindsightClient.listMemories(bankId, {
      limit,
      offset,
      type,
      q,
      consolidationState,
    });

    return NextResponse.json(response, { status: 200 });
  } catch (error) {
    console.error("Error listing memory units:", error);
    return NextResponse.json({ error: "Failed to list memory units" }, { status: 500 });
  }
}

// Note: Individual memory unit deletion is not yet supported by the API
// Use clearBankMemories to delete all memories for a bank instead
export async function DELETE(request: NextRequest) {
  return NextResponse.json(
    {
      error:
        "Individual memory unit deletion is not yet supported. Use clear all memories instead.",
    },
    { status: 501 } // Not Implemented
  );
}
