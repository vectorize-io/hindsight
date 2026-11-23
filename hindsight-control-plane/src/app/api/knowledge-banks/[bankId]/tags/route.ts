import { NextRequest } from "next/server";
import { lowLevelClient, sdk } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ bankId: string }> }
) {
  const { bankId } = await params;
  const sp = request.nextUrl.searchParams;
  const response = await sdk.listKnowledgeBankTags({
    client: lowLevelClient,
    path: { bank_id: bankId },
    query: {
      q: sp.get("q") ?? undefined,
      limit: sp.has("limit") ? Number(sp.get("limit")) : undefined,
      offset: sp.has("offset") ? Number(sp.get("offset")) : undefined,
    },
  });
  return respondWithSdk(response, "Failed to list knowledge bank tags", { request });
}
