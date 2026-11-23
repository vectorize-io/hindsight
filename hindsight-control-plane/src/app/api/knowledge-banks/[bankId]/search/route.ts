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
  const response = await sdk.searchKnowledgeBank({
    client: lowLevelClient,
    path: { bank_id: bankId },
    body,
  });
  // A 503 here means the knowledge embedder is unreachable: the search panel shows why.
  return respondWithSdk(response, "Failed to search knowledge bank", {
    request,
    exposeUnavailable: true,
  });
}
