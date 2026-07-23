import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { sdk, getDataplaneClient } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

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

  const response = await sdk.listDocuments({
    client: getDataplaneClient(request),
    path: { bank_id: bankId },
    query: { q, limit, offset },
  });
  return respondWithSdk(response, "Failed to fetch documents", { request });
}
