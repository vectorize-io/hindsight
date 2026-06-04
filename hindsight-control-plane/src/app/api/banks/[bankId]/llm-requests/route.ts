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

    // Forward query params
    const { searchParams } = new URL(request.url);
    const query = searchParams.toString();

    const url = dataplaneBankUrl(bankId, `/llm-requests${query ? `?${query}` : ""}`);
    const response = await fetch(url, {
      method: "GET",
      headers: getDataplaneHeaders(),
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: data.detail || "Failed to list LLM requests",
          errorKey: "api.errors.llmRequests.list",
        }),
        { status: response.status }
      );
    }

    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error listing LLM requests:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to list LLM requests",
        errorKey: "api.errors.llmRequests.list",
      }),
      { status: 500 }
    );
  }
}
