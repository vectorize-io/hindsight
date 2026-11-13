import { NextRequest } from "next/server";
import { missingQueryParam } from "@/lib/i18n/api-errors";
import { sdk, lowLevelClient } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";
import { readTagFilter } from "@/lib/tag-filter-params";

export async function GET(request: NextRequest) {
  const bankId = request.nextUrl.searchParams.get("bank_id");
  if (!bankId) return missingQueryParam(request, "bank_id");

  const searchParams = request.nextUrl.searchParams;
  const response = await sdk.getObservationTree({
    client: lowLevelClient,
    path: { bank_id: bankId },
    query: {
      ...readTagFilter(searchParams),
      tag_groups: searchParams.get("tag_groups") ?? undefined,
    },
  });
  return respondWithSdk(response, "Failed to get observation tree", { request });
}
