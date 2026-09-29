import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ memoryId: string }> }
) {
  try {
    const { memoryId } = await params;
    const searchParams = request.nextUrl.searchParams;
    const bankId = searchParams.get("bank_id");

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
      dataplaneBankUrl(bankId, `/memories/${encodeURIComponent(memoryId)}/history`),
      {
        method: "GET",
        headers: getDataplaneHeaders({ "Content-Type": "application/json" }),
      }
    );

    if (!response.ok) {
      if (response.status === 404) {
        return NextResponse.json(
          localizeApiErrorPayload(request, {
            error: "Memory not found",
            errorKey: "api.errors.memories.notFound",
          }),
          { status: 404 }
        );
      }
      throw new Error(`API returned ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error fetching observation history:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to fetch observation history",
        errorKey: "api.errors.memories.history",
      }),
      { status: 500 }
    );
  }
}
