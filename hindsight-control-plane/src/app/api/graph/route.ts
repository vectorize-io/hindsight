import { NextRequest, NextResponse } from "next/server";
import { sdk, lowLevelClient } from "@/lib/hindsight-client";

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const bankId = searchParams.get("bank_id") || searchParams.get("agent_id");

    if (!bankId) {
      return NextResponse.json({ error: "bank_id is required" }, { status: 400 });
    }

    // Get optional query parameters
    const type = searchParams.get("type") || searchParams.get("fact_type") || undefined;
    const limitParam = searchParams.get("limit");
    const limit = limitParam ? parseInt(limitParam, 10) : undefined;

    const response = await sdk.getGraph({
      client: lowLevelClient,
      path: { bank_id: bankId },
      query: {
        type: type,
        limit: limit,
      },
    });

    if (response.error || !response.data) {
      console.error("Graph API error:", response.error);
      return NextResponse.json(
        { error: response.error || "Failed to fetch graph data" },
        { status: 500 }
      );
    }

    return NextResponse.json(response.data, { status: 200 });
  } catch (error) {
    console.error("Error fetching graph data:", error);
    return NextResponse.json({ error: "Failed to fetch graph data" }, { status: 500 });
  }
}
