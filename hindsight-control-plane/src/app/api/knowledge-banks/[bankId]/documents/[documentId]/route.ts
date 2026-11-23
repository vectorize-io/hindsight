import { NextRequest } from "next/server";
import { lowLevelClient, sdk } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

type Params = { params: Promise<{ bankId: string; documentId: string }> };

export async function GET(request: NextRequest, { params }: Params) {
  const { bankId, documentId } = await params;
  const response = await sdk.getKnowledgeDocument({
    client: lowLevelClient,
    path: { bank_id: bankId, document_id: documentId },
  });
  return respondWithSdk(response, "Failed to fetch knowledge document", { request });
}

export async function DELETE(request: NextRequest, { params }: Params) {
  const { bankId, documentId } = await params;
  const response = await sdk.deleteKnowledgeDocument({
    client: lowLevelClient,
    path: { bank_id: bankId, document_id: documentId },
  });
  return respondWithSdk(response, "Failed to delete knowledge document", { request });
}
