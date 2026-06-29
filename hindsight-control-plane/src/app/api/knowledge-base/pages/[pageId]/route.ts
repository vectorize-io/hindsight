import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ pageId: string }> }
) {
  try {
    const { pageId } = await params;
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
    const response = await fetch(
      dataplaneBankUrl(
        bankId,
        `/knowledge-base/pages/${encodeURIComponent(decodeURIComponent(pageId))}`
      ),
      { headers: getDataplaneHeaders() }
    );
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      return NextResponse.json(error, { status: response.status });
    }
    return NextResponse.json(await response.json(), { status: 200 });
  } catch (error) {
    console.error("Failed to fetch knowledge page:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to fetch knowledge page",
        errorKey: "api.errors.knowledgeBase.fetchPage",
      }),
      { status: 500 }
    );
  }
}
