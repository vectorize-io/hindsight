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
      dataplaneBankUrl(bankId, `/memories/${encodeURIComponent(memoryId)}`),
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
    console.error("Error fetching memory:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to fetch memory",
        errorKey: "api.errors.memories.fetch",
      }),
      { status: 500 }
    );
  }
}

export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ memoryId: string }> }
) {
  try {
    const { memoryId } = await params;
    const body = await request.json();
    const bankId = body.bank_id || request.nextUrl.searchParams.get("bank_id");

    if (!bankId) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "bank_id is required",
          errorKey: "api.errors.validation.bankIdRequired",
        }),
        { status: 400 }
      );
    }

    // Curation fields only; bank_id is a routing param, not part of the body.
    const { text, context, occurred_start, occurred_end, fact_type, entities, state, reason } =
      body;

    const response = await fetch(
      dataplaneBankUrl(bankId, `/memories/${encodeURIComponent(memoryId)}`),
      {
        method: "PATCH",
        headers: getDataplaneHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({
          text,
          context,
          occurred_start,
          occurred_end,
          fact_type,
          entities,
          state,
          reason,
        }),
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
      const detail = await response.text();
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: detail || `API returned ${response.status}`,
          errorKey: "api.errors.memories.update",
        }),
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error updating memory:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to update memory",
        errorKey: "api.errors.memories.update",
      }),
      { status: 500 }
    );
  }
}
