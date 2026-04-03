import { NextRequest, NextResponse } from "next/server";
import { sdk, getClientForTenant } from "@/lib/hindsight-client";

export async function GET(request: NextRequest) {
  try {
    const tenant = request.nextUrl.searchParams.get("tenant");
    const { lowLevelClient } = getClientForTenant(tenant);
    const response = await sdk.getVersion({
      client: lowLevelClient,
    });

    if (response.error) {
      console.error("API error getting version:", response.error);
      return NextResponse.json({ error: "Failed to get version" }, { status: 500 });
    }

    return NextResponse.json(response.data, { status: 200 });
  } catch (error) {
    console.error("Error getting version:", error);
    return NextResponse.json({ error: "Failed to get version" }, { status: 500 });
  }
}
