import { NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { sdk, createDataplaneClientForRequest } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

export async function POST(request: Request, { params }: { params: Promise<{ bankId: string }> }) {
  const { bankId } = await params;

  if (!bankId) {
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "bank_id is required",
        errorKey: "api.errors.validation.bankIdRequired",
      }),
      { status: 400 }
    );
  }

  const response = await sdk.recoverConsolidation({
    client: createDataplaneClientForRequest(request),
    path: { bank_id: bankId },
  });
  return respondWithSdk(response, "Failed to recover consolidation", { request });
}
