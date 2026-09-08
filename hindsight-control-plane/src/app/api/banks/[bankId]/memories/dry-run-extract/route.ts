import { NextRequest } from "next/server";
import { lowLevelClient, sdk } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

/**
 * Proxy for the dataplane dry-run extraction endpoint: extract facts from sample
 * text without storing anything. Bank-scoped, so the prompt tester can reach it the
 * same way it reaches the prompt preview.
 */
export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ bankId: string }> }
) {
  const { bankId } = await params;
  const body = await request.json();
  const response = await sdk.dryRunExtractMemories({
    client: lowLevelClient,
    path: { bank_id: bankId },
    body,
  });
  return respondWithSdk(response, "Failed to run dry-run extraction", { request });
}
