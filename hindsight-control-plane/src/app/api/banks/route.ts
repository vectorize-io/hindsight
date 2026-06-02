import { NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { sdk, lowLevelClient } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

const HTTP_CREATED = 201;

export async function GET(request: Request) {
  const response = await sdk.listBanks({ client: lowLevelClient });
  return respondWithSdk(response, "Failed to fetch banks", { request });
}

export async function POST(request: Request) {
  let body;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Invalid JSON body",
        errorKey: "api.errors.auth.invalidRequestBody",
      }),
      { status: 400 }
    );
  }
  const { bank_id } = body;

  if (!bank_id) {
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "bank_id is required",
        errorKey: "api.errors.validation.bankIdRequired",
      }),
      { status: 400 }
    );
  }

  const response = await sdk.createOrUpdateBank({
    client: lowLevelClient,
    path: { bank_id },
    body: {},
  });
  return respondWithSdk(response, "Failed to create bank", HTTP_CREATED, { request });
}
