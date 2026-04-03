import { NextRequest, NextResponse } from "next/server";
import { sdk, getClientForTenant } from "@/lib/hindsight-client";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ agentId: string }> }
) {
  try {
    const tenant = request.nextUrl.searchParams.get("tenant");
    const { lowLevelClient } = getClientForTenant(tenant);
    const { agentId } = await params;
    const response = await sdk.getAgentStats({
      client: lowLevelClient,
      path: { bank_id: agentId },
    });
    return NextResponse.json(response.data, { status: 200 });
  } catch (error) {
    console.error("Error fetching stats:", error);
    return NextResponse.json({ error: "Failed to fetch stats" }, { status: 500 });
  }
}
