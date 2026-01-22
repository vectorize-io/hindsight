import { NextRequest, NextResponse } from "next/server";
import { sdk, lowLevelClient } from "@/lib/hindsight-client";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ agentId: string }> }
) {
  try {
    const { agentId } = await params;
    const searchParams = request.nextUrl.searchParams;
    const status = searchParams.get("status") || undefined;
    const limit = searchParams.get("limit") ? parseInt(searchParams.get("limit")!) : undefined;
    const offset = searchParams.get("offset") ? parseInt(searchParams.get("offset")!) : undefined;

    const response = await sdk.listOperations({
      client: lowLevelClient,
      path: { bank_id: agentId },
      query: { status, limit, offset },
    });
    return NextResponse.json(response.data || {}, { status: 200 });
  } catch (error) {
    console.error("Error fetching operations:", error);
    return NextResponse.json({ error: "Failed to fetch operations" }, { status: 500 });
  }
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ agentId: string }> }
) {
  try {
    const { agentId } = await params;
    const searchParams = request.nextUrl.searchParams;
    const operationId = searchParams.get("operation_id");

    if (!operationId) {
      return NextResponse.json({ error: "operation_id is required" }, { status: 400 });
    }

    const response = await sdk.cancelOperation({
      client: lowLevelClient,
      path: { bank_id: agentId, operation_id: operationId },
    });

    return NextResponse.json(response.data || {}, { status: 200 });
  } catch (error) {
    console.error("Error canceling operation:", error);
    return NextResponse.json({ error: "Failed to cancel operation" }, { status: 500 });
  }
}
