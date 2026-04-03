import { NextRequest, NextResponse } from "next/server";
import { sdk, getClientForTenant } from "@/lib/hindsight-client";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ chunkId: string }> }
) {
  try {
    const tenant = request.nextUrl.searchParams.get("tenant");
    const { lowLevelClient } = getClientForTenant(tenant);
    const { chunkId } = await params;

    const response = await sdk.getChunk({
      client: lowLevelClient,
      path: { chunk_id: chunkId },
    });

    return NextResponse.json(response.data, { status: 200 });
  } catch (error) {
    console.error("Error fetching chunk:", error);
    return NextResponse.json({ error: "Failed to fetch chunk" }, { status: 500 });
  }
}
