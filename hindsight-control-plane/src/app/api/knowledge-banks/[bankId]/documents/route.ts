import { NextRequest } from "next/server";
import { lowLevelClient, sdk } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ bankId: string }> }
) {
  const { bankId } = await params;
  const sp = request.nextUrl.searchParams;
  const response = await sdk.listKnowledgeDocuments({
    client: lowLevelClient,
    path: { bank_id: bankId },
    query: {
      limit: sp.has("limit") ? Number(sp.get("limit")) : undefined,
      offset: sp.has("offset") ? Number(sp.get("offset")) : undefined,
      q: sp.get("q") ?? undefined,
      status: sp.get("status") ?? undefined,
      media_type: sp.get("media_type") ?? undefined,
      tags: sp.has("tags") ? sp.getAll("tags") : undefined,
    },
  });
  return respondWithSdk(response, "Failed to list knowledge documents", { request });
}
