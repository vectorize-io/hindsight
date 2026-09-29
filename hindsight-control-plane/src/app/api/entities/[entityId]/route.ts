import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { sdk, lowLevelClient } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ entityId: string }> }
) {
  const { entityId } = await params;
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

  // Decode URL-encoded entityId in case it contains special chars
  const decodedEntityId = decodeURIComponent(entityId);

  const response = await sdk.getEntity({
    client: lowLevelClient,
    path: {
      bank_id: bankId,
      entity_id: decodedEntityId,
    },
  });
  return respondWithSdk(response, "Failed to get entity", { request });
}
