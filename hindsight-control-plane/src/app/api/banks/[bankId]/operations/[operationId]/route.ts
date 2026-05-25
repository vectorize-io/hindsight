import { NextResponse } from "next/server";
import { sdk, lowLevelClient, dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

export async function GET(
  request: Request,
  { params }: { params: Promise<{ bankId: string; operationId: string }> }
) {
  const { bankId, operationId } = await params;

  if (!bankId) {
    return NextResponse.json({ error: "bank_id is required" }, { status: 400 });
  }

  if (!operationId) {
    return NextResponse.json({ error: "operation_id is required" }, { status: 400 });
  }

  const url = new URL(request.url);
  const includePayload = url.searchParams.get("include_payload") === "true";

  const response = await sdk.getOperationStatus({
    client: lowLevelClient,
    path: { bank_id: bankId, operation_id: operationId },
    query: includePayload ? { include_payload: true } : undefined,
  });
  return respondWithSdk(response, "Failed to get operation status");
}

export async function POST(
  request: Request,
  { params }: { params: Promise<{ bankId: string; operationId: string }> }
) {
  try {
    const { bankId, operationId } = await params;

    if (!bankId) {
      return NextResponse.json({ error: "bank_id is required" }, { status: 400 });
    }

    if (!operationId) {
      return NextResponse.json({ error: "operation_id is required" }, { status: 400 });
    }

    const url = dataplaneBankUrl(bankId, `/operations/${encodeURIComponent(operationId)}/retry`);
    const response = await fetch(url, {
      method: "POST",
      headers: getDataplaneHeaders({ "Content-Type": "application/json" }),
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        { error: data.detail || "Failed to retry operation" },
        { status: response.status }
      );
    }

    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error retrying operation:", error);
    return NextResponse.json({ error: "Failed to retry operation" }, { status: 500 });
  }
}
