import { NextRequest } from "next/server";
import { lowLevelClient, sdk } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

type Query = NonNullable<Parameters<typeof sdk.getKnowledgeBankTimeseries>[0]["query"]>;

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ bankId: string }> }
) {
  const { bankId } = await params;
  const sp = request.nextUrl.searchParams;
  const response = await sdk.getKnowledgeBankTimeseries({
    client: lowLevelClient,
    path: { bank_id: bankId },
    query: {
      period: (sp.get("period") ?? undefined) as Query["period"],
      media_type: (sp.get("media_type") ?? undefined) as Query["media_type"],
    },
  });
  return respondWithSdk(response, "Failed to fetch knowledge bank activity", { request });
}
