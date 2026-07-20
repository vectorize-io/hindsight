import { NextRequest } from "next/server";
import { lowLevelClient, sdk } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

export async function POST(request: NextRequest) {
  const response = await sdk.testServerLlm({ client: lowLevelClient });
  return respondWithSdk(response, "Failed to test server LLM connectivity", { request });
}
