import { NextRequest, NextResponse } from "next/server";
import { lowLevelClient, sdk } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ bankId: string }> }
) {
  const { bankId } = await params;
  const body = await request.json().catch(() => null);
  if (body === null) return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  const response = await sdk.ingestKnowledgeTexts({
    client: lowLevelClient,
    path: { bank_id: bankId },
    body,
  });
  // 200 when processed in the request, 202 when queued.
  return respondWithSdk(
    response,
    "Failed to ingest knowledge texts",
    response.response?.status ?? 200,
    {
      request,
    }
  );
}
