import { NextRequest } from "next/server";
import { lowLevelClient, sdk } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

type Params = { params: Promise<{ bankId: string; operationId: string }> };

export async function GET(request: NextRequest, { params }: Params) {
  const { bankId, operationId } = await params;
  const includePayload = request.nextUrl.searchParams.get("include_payload") === "true";
  const response = await sdk.getKnowledgeBankOperation({
    client: lowLevelClient,
    path: { bank_id: bankId, operation_id: operationId },
    query: includePayload ? { include_payload: true } : undefined,
  });
  return respondWithSdk(response, "Failed to get knowledge bank operation", { request });
}

/** Cancels a pending or running operation. */
export async function DELETE(request: NextRequest, { params }: Params) {
  const { bankId, operationId } = await params;
  const response = await sdk.cancelKnowledgeBankOperation({
    client: lowLevelClient,
    path: { bank_id: bankId, operation_id: operationId },
  });
  return respondWithSdk(response, "Failed to cancel knowledge bank operation", { request });
}
