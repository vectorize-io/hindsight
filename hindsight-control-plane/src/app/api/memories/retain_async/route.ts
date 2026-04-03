import { NextRequest, NextResponse } from "next/server";
import { sdk, getClientForTenant } from "@/lib/hindsight-client";

export async function POST(request: NextRequest) {
  try {
    const tenant = request.nextUrl.searchParams.get("tenant");
    const { lowLevelClient } = getClientForTenant(tenant);
    const body = await request.json();
    const bankId = body.bank_id || body.agent_id;

    if (!bankId) {
      return NextResponse.json({ error: "bank_id is required" }, { status: 400 });
    }

    const { items } = body;

    const response = await sdk.retainMemories({
      client: lowLevelClient,
      path: { bank_id: bankId },
      body: { items, async: true },
    });

    return NextResponse.json(response.data, { status: 200 });
  } catch (error) {
    console.error("Error batch retain async:", error);
    return NextResponse.json({ error: "Failed to batch retain async" }, { status: 500 });
  }
}
