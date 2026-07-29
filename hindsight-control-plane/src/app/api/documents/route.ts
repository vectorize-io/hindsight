import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { sdk, lowLevelClient } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

// Tag matching modes accepted by the dataplane's list_documents endpoint.
const TAGS_MATCH_MODES = new Set(["any", "all", "any_strict", "all_strict", "exact"]);

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const bankId = searchParams.get("bank_id");

  if (!bankId) {
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "bank_id is required",
        errorKey: "api.errors.validation.bankIdRequired",
      }),
      { status: 400 }
    );
  }

  const q = searchParams.get("q") || undefined;
  const limit = searchParams.get("limit") ? Number(searchParams.get("limit")) : undefined;
  const offset = searchParams.get("offset") ? Number(searchParams.get("offset")) : undefined;
  const tagList = searchParams.getAll("tags").filter((tag) => tag.length > 0);
  const tags = tagList.length > 0 ? tagList : undefined;
  // Only forward tags_match alongside tags — on its own it would override the
  // dataplane default for an unfiltered listing.
  const tagsMatchParam = searchParams.get("tags_match");
  const tagsMatch =
    tags && tagsMatchParam && TAGS_MATCH_MODES.has(tagsMatchParam) ? tagsMatchParam : undefined;

  const response = await sdk.listDocuments({
    client: lowLevelClient,
    path: { bank_id: bankId },
    query: { q, tags, tags_match: tagsMatch, limit, offset },
  });
  return respondWithSdk(response, "Failed to fetch documents", { request });
}
