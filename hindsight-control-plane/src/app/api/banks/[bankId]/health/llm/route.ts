import { NextRequest } from "next/server";
import { createDataplaneClientForRequest, sdk } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ bankId: string }> }
) {
  const { bankId } = await params;
  const response = await sdk.testBankLlm({
    client: createDataplaneClientForRequest(request),
    path: { bank_id: bankId },
  });
  return respondWithSdk(response, "Failed to test bank LLM connectivity", { request });
}
