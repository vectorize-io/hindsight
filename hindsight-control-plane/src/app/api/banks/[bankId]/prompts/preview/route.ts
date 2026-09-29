import { NextRequest } from "next/server";
import { lowLevelClient, sdk } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

/**
 * Proxy for the dataplane prompt-preview endpoint: render the messages an operation
 * would send for this bank, without calling an LLM or changing anything. The body
 * carries the mission (and other prompt-affecting settings) the caller is editing, so
 * an unsaved change can be previewed before it is saved.
 */
export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ bankId: string }> }
) {
  const { bankId } = await params;
  const body = await request.json();
  const response = await sdk.previewPrompt({
    client: lowLevelClient,
    path: { bank_id: bankId },
    body,
  });
  return respondWithSdk(response, "Failed to preview prompt", { request });
}
