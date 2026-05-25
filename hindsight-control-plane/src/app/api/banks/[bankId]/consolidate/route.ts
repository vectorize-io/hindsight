import { NextResponse } from "next/server";
import { sdk, lowLevelClient } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

export async function POST(request: Request, { params }: { params: Promise<{ bankId: string }> }) {
  const { bankId } = await params;

  if (!bankId) {
    return NextResponse.json({ error: "bank_id is required" }, { status: 400 });
  }

  const response = await sdk.triggerConsolidation({
    client: lowLevelClient,
    path: { bank_id: bankId },
  });
  return respondWithSdk(response, "Failed to trigger consolidation");
}
