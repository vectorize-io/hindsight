import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

export async function GET(request: NextRequest) {
  try {
    const bankId = request.nextUrl.searchParams.get("bank_id");
    const q = request.nextUrl.searchParams.get("q");
    const limit = request.nextUrl.searchParams.get("limit") ?? "10";
    if (!bankId) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "bank_id is required",
          errorKey: "api.errors.validation.bankIdRequired",
        }),
        { status: 400 }
      );
    }
    if (!q || !q.trim()) {
      return NextResponse.json({ results: [], total: 0 }, { status: 200 });
    }
    const path = `/knowledge-base/search?q=${encodeURIComponent(q)}&limit=${encodeURIComponent(limit)}`;
    const response = await fetch(dataplaneBankUrl(bankId, path), {
      headers: getDataplaneHeaders(),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      return NextResponse.json(error, { status: response.status });
    }
    return NextResponse.json(await response.json(), { status: 200 });
  } catch (error) {
    console.error("Failed to search knowledge base:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to search knowledge base",
        errorKey: "api.errors.knowledgeBase.search",
      }),
      { status: 500 }
    );
  }
}
