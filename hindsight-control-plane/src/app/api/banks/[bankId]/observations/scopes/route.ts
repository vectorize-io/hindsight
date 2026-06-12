import { NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

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

    const { searchParams } = new URL(request.url);
    const limit = searchParams.get("limit");

    const queryParams = new URLSearchParams();
    if (limit) queryParams.append("limit", limit);

    const url = dataplaneBankUrl(
      bankId,
      `/observations/scopes${queryParams.toString() ? `?${queryParams}` : ""}`
    );
    const response = await fetch(url, { method: "GET", headers: getDataplaneHeaders() });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("API error listing observation scopes:", errorText);
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "Failed to list observation scopes",
          errorKey: "api.errors.observations.scopes",
        }),
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error listing observation scopes:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to list observation scopes",
        errorKey: "api.errors.observations.scopes",
      }),
      { status: 500 }
    );
  }
}
