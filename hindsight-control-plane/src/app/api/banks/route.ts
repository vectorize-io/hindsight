import { NextRequest, NextResponse } from "next/server";
import { sdk, getClientForTenant } from "@/lib/hindsight-client";

export async function GET(request: NextRequest) {
  try {
    const tenant = request.nextUrl.searchParams.get("tenant");
    const { lowLevelClient } = getClientForTenant(tenant);
    const response = await sdk.listBanks({ client: lowLevelClient });

    // Check if the response has an error or no data
    if (response.error || !response.data) {
      console.error("API error:", response.error);
      return NextResponse.json({ error: "Failed to fetch banks from API" }, { status: 500 });
    }

    return NextResponse.json(response.data, { status: 200 });
  } catch (error) {
    console.error("Error fetching banks:", error);
    return NextResponse.json({ error: "Failed to fetch banks" }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const tenant = request.nextUrl.searchParams.get("tenant");
    const { lowLevelClient } = getClientForTenant(tenant);
    const body = await request.json();
    const { bank_id } = body;

    if (!bank_id) {
      return NextResponse.json({ error: "bank_id is required" }, { status: 400 });
    }

    const response = await sdk.createOrUpdateBank({
      client: lowLevelClient,
      path: { bank_id },
      body: {},
    });

    const serializedData = JSON.parse(JSON.stringify(response.data));
    return NextResponse.json(serializedData, { status: 201 });
  } catch (error) {
    console.error("Error creating bank:", error);
    return NextResponse.json({ error: "Failed to create bank" }, { status: 500 });
  }
}
