import { NextRequest } from "next/server";
import { missingQueryParam } from "@/lib/i18n/api-errors";
import { sdk, lowLevelClient } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

export async function GET(request: NextRequest) {
  const bankId = request.nextUrl.searchParams.get("bank_id");
  if (!bankId) return missingQueryParam(request, "bank_id");

  const response = await sdk.getMemoryTree({
    client: lowLevelClient,
    path: { bank_id: bankId },
  });
  return respondWithSdk(response, "Failed to get memory tree", { request });
}
