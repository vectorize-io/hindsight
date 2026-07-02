import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

export async function POST(request: NextRequest) {
  try {
    const bankId = request.nextUrl.searchParams.get("bank_id");
    if (!bankId) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "bank_id is required",
          errorKey: "api.errors.validation.bankIdRequired",
        }),
        { status: 400 }
      );
    }
    const body = await request.json();
    const response = await fetch(dataplaneBankUrl(bankId, "/knowledge-base/pages"), {
      method: "POST",
      headers: getDataplaneHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      return NextResponse.json(error, { status: response.status });
    }
    return NextResponse.json(await response.json(), { status: response.status });
  } catch (error) {
    console.error("Failed to create page:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to create page",
        errorKey: "api.errors.knowledgeBase.createPage",
      }),
      { status: 500 }
    );
  }
}
