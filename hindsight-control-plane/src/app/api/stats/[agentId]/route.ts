import { NextRequest } from "next/server";
import { sdk, createDataplaneClientForRequest } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ agentId: string }> }
) {
  const { agentId } = await params;
  const response = await sdk.getAgentStats({
    client: createDataplaneClientForRequest(request),
    path: { bank_id: agentId },
  });
  return respondWithSdk(response, "Failed to fetch stats", { request });
}
