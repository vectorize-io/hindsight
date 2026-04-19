import { NextRequest, NextResponse } from "next/server";
import { sdk, lowLevelClient } from "@/lib/hindsight-client";

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const bankId = searchParams.get("bank_id");

    if (!bankId) {
      return NextResponse.json({ error: "bank_id is required" }, { status: 400 });
    }

    const limit = searchParams.get("limit") ? Number(searchParams.get("limit")) : undefined;
    const minCount = searchParams.get("min_count")
      ? Number(searchParams.get("min_count"))
      : undefined;

    const response = await sdk.getEntityGraph({
      client: lowLevelClient,
      path: { bank_id: bankId },
      query: { limit, min_count: minCount },
    });

    if (response.error || !response.data) {
      console.error("Entity graph API error:", response.error);
      return NextResponse.json(
        { error: response.error || "Failed to fetch entity graph" },
        { status: 500 }
      );
    }

    return NextResponse.json(response.data, { status: 200 });
  } catch (error) {
    console.error("Error fetching entity graph:", error);
    return NextResponse.json({ error: "Failed to fetch entity graph" }, { status: 500 });
  }
}
