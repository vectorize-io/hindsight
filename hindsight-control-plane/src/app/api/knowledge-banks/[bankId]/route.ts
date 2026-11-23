import { NextRequest } from "next/server";
import { lowLevelClient, sdk } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

type Params = { params: Promise<{ bankId: string }> };

export async function GET(request: NextRequest, { params }: Params) {
  const { bankId } = await params;
  const response = await sdk.getKnowledgeBank({
    client: lowLevelClient,
    path: { bank_id: bankId },
  });
  return respondWithSdk(response, "Failed to fetch knowledge bank", { request });
}

export async function DELETE(request: NextRequest, { params }: Params) {
  const { bankId } = await params;
  const response = await sdk.deleteKnowledgeBank({
    client: lowLevelClient,
    path: { bank_id: bankId },
  });
  return respondWithSdk(response, "Failed to delete knowledge bank", { request });
}
