import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload, missingQueryParam } from "@/lib/i18n/api-errors";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

export async function POST(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const bankId = searchParams.get("bank_id");
    const documentId = searchParams.get("document_id");

    if (!bankId) return missingQueryParam(request, "bank_id");
    if (!documentId) return missingQueryParam(request, "document_id");

    const response = await fetch(
      dataplaneBankUrl(bankId, `/documents/${encodeURIComponent(documentId)}/reprocess`),
      {
        method: "POST",
        headers: getDataplaneHeaders({ "Content-Type": "application/json" }),
      }
    );

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      return NextResponse.json(error, { status: response.status });
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error reprocessing document:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to reprocess document",
        errorKey: "api.errors.documents.reprocess",
      }),
      { status: 500 }
    );
  }
}
