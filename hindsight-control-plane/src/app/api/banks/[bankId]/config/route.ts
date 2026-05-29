import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { lowLevelClient, sdk } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ bankId: string }> }
) {
  const { bankId } = await params;
  const response = await sdk.getBankConfig({
    client: lowLevelClient,
    path: { bank_id: bankId },
  });
  return respondWithSdk(response, "Failed to fetch bank config", { request });
}

export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ bankId: string }> }
) {
  const { bankId } = await params;
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
  const { updates } = body;

  const response = await sdk.updateBankConfig({
    client: lowLevelClient,
    path: { bank_id: bankId },
    body: { updates },
  });
  return respondWithSdk(response, "Failed to update bank config", { request });
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ bankId: string }> }
) {
  const { bankId } = await params;
  const response = await sdk.resetBankConfig({
    client: lowLevelClient,
    path: { bank_id: bankId },
  });
  return respondWithSdk(response, "Failed to reset bank config", { request });
}
