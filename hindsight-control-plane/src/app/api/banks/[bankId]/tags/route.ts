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
    const q = searchParams.get("q");
    const source = searchParams.get("source");
    const limit = searchParams.get("limit");
    const offset = searchParams.get("offset");

    const queryParams = new URLSearchParams();
    if (q) queryParams.append("q", q);
    if (source) queryParams.append("source", source);
    if (limit) queryParams.append("limit", limit);
    if (offset) queryParams.append("offset", offset);

    const url = dataplaneBankUrl(bankId, `/tags${queryParams.toString() ? `?${queryParams}` : ""}`);
    const response = await fetch(url, { method: "GET", headers: getDataplaneHeaders() });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("API error listing tags:", errorText);
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "Failed to list tags",
          errorKey: "api.errors.tags.list",
        }),
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error listing tags:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to list tags",
        errorKey: "api.errors.tags.list",
      }),
      { status: 500 }
    );
  }
}
