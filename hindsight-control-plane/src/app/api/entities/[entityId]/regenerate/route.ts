import { NextRequest, NextResponse } from "next/server";
import { sdk, lowLevelClient } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ entityId: string }> }
) {
  const { entityId } = await params;
  const searchParams = request.nextUrl.searchParams;
  const bankId = searchParams.get("bank_id");

  if (!bankId) {
    return NextResponse.json({ error: "bank_id is required" }, { status: 400 });
  }

  const decodedEntityId = decodeURIComponent(entityId);

  const response = await sdk.regenerateEntityObservations({
    client: lowLevelClient,
    path: {
      bank_id: bankId,
      entity_id: decodedEntityId,
    },
  });
  return respondWithSdk(response, "Failed to regenerate entity observations");
}
