import { NextRequest } from "next/server";
import { sdk, lowLevelClient } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";
import { getSessionPrefix } from "@/lib/auth/session";
import { bankAllowed } from "@/lib/auth/tokens";
import { forbiddenResponse } from "@/lib/auth/bank-guard";

/**
 * Fetch a chunk by id. Chunks are addressed by a global chunk id, not a
 * bank-scoped path, so middleware can't gate this route by prefix up front. The
 * chunk's owning bank only becomes known in the response, so a scoped session is
 * enforced here: fetch, then 403 if the chunk belongs to a bank outside the
 * session's prefix. Admin sessions (null prefix) and access-key-less setups pass
 * straight through.
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ chunkId: string }> }
) {
  const { chunkId } = await params;
  const response = await sdk.getChunk({
    client: lowLevelClient,
    path: { chunk_id: chunkId },
  });

  const prefix = await getSessionPrefix(request);
  if (prefix) {
    const bankId = (response.data as { bank_id?: string } | undefined)?.bank_id;
    if (bankId !== undefined && !bankAllowed(prefix, bankId)) {
      return forbiddenResponse(request);
    }
  }

  return respondWithSdk(response, "Failed to fetch chunk", { request });
}
