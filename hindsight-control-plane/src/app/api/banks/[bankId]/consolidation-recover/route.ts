import { NextRequest, NextResponse } from "next/server";
import { sdk, getClientForTenant } from "@/lib/hindsight-client";

export async function POST(request: NextRequest, { params }: { params: Promise<{ bankId: string }> }) {
  try {
    const tenant = request.nextUrl.searchParams.get("tenant");
    const { lowLevelClient } = getClientForTenant(tenant);
    const { bankId } = await params;

    if (!bankId) {
      return NextResponse.json({ error: "bank_id is required" }, { status: 400 });
    }

    const response = await sdk.recoverConsolidation({
      client: lowLevelClient,
      path: { bank_id: bankId },
    });

    if (response.error) {
      console.error("API error recovering consolidation:", response.error);
      return NextResponse.json({ error: "Failed to recover consolidation" }, { status: 500 });
    }

    return NextResponse.json(response.data, { status: 200 });
  } catch (error) {
    console.error("Error recovering consolidation:", error);
    return NextResponse.json({ error: "Failed to recover consolidation" }, { status: 500 });
  }
}
