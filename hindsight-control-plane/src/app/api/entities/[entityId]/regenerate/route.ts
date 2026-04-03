import { NextRequest, NextResponse } from "next/server";
import { sdk, getClientForTenant } from "@/lib/hindsight-client";

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ entityId: string }> }
) {
  try {
    const tenant = request.nextUrl.searchParams.get("tenant");
    const { lowLevelClient } = getClientForTenant(tenant);
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

    return NextResponse.json(response.data, { status: 200 });
  } catch (error) {
    console.error("Error regenerating entity observations:", error);
    return NextResponse.json(
      { error: "Failed to regenerate entity observations" },
      { status: 500 }
    );
  }
}
