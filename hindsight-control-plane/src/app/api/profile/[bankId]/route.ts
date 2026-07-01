import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { sdk, createDataplaneClientForRequest } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ bankId: string }> }
) {
  const { bankId } = await params;
  const response = await sdk.getBankProfile({
    client: createDataplaneClientForRequest(request),
    path: { bank_id: bankId },
  });
  return respondWithSdk(response, "Failed to fetch bank profile", { request });
}

export async function PUT(
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

  const response = await sdk.createOrUpdateBank({
    client: createDataplaneClientForRequest(request),
    path: { bank_id: bankId },
    body: body,
  });
  return respondWithSdk(response, "Failed to update bank profile", { request });
}
