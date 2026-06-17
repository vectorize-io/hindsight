import { NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { DATAPLANE_URL, getDataplaneHeaders } from "@/lib/hindsight-client";

export async function GET(request: Request, { params }: { params: Promise<{ bankId: string }> }) {
  try {
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

    const response = await fetch(
      `${DATAPLANE_URL}/v1/default/banks/${bankId}/observations/scopes`,
      { headers: getDataplaneHeaders() }
    );

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      return NextResponse.json(error, { status: response.status });
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error listing observation scopes:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to list observation scopes",
        errorKey: "api.errors.observations.list",
      }),
      { status: 500 }
    );
  }
}
