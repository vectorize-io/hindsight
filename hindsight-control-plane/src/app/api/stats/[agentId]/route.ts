import { NextRequest } from "next/server";
import { sdk, getDataplaneClient } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ agentId: string }> }
) {
  const { agentId } = await params;
  const response = await sdk.getAgentStats({
    client: getDataplaneClient(request),
    path: { bank_id: agentId },
  });
  return respondWithSdk(response, "Failed to fetch stats", { request });
}
