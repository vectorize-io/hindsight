import { NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

export async function GET(
  request: Request,
  { params }: { params: Promise<{ bankId: string; directiveId: string }> }
) {
  try {
    const { bankId, directiveId } = await params;

    if (!bankId || !directiveId) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "bank_id and directive_id are required",
          errorKey: "api.errors.validation.bankAndDirectiveIdRequired",
        }),
        { status: 400 }
      );
    }

    const response = await fetch(
      dataplaneBankUrl(bankId, `/directives/${encodeURIComponent(directiveId)}`),
      { method: "GET", headers: getDataplaneHeaders() }
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error("API error getting directive:", errorText);
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "Failed to get directive",
          errorKey: "api.errors.directives.fetch",
        }),
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error getting directive:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to get directive",
        errorKey: "api.errors.directives.fetch",
      }),
      { status: 500 }
    );
  }
}

export async function PATCH(
  request: Request,
  { params }: { params: Promise<{ bankId: string; directiveId: string }> }
) {
  try {
    const { bankId, directiveId } = await params;

    if (!bankId || !directiveId) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "bank_id and directive_id are required",
          errorKey: "api.errors.validation.bankAndDirectiveIdRequired",
        }),
        { status: 400 }
      );
    }

    const body = await request.json();

    const response = await fetch(
      dataplaneBankUrl(bankId, `/directives/${encodeURIComponent(directiveId)}`),
      {
        method: "PATCH",
        headers: getDataplaneHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(body),
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error("API error updating directive:", errorText);
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: errorText || "Failed to update directive",
          errorKey: "api.errors.directives.update",
        }),
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error updating directive:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to update directive",
        errorKey: "api.errors.directives.update",
      }),
      { status: 500 }
    );
  }
}

export async function DELETE(
  request: Request,
  { params }: { params: Promise<{ bankId: string; directiveId: string }> }
) {
  try {
    const { bankId, directiveId } = await params;

    if (!bankId || !directiveId) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "bank_id and directive_id are required",
          errorKey: "api.errors.validation.bankAndDirectiveIdRequired",
        }),
        { status: 400 }
      );
    }

    const response = await fetch(
      dataplaneBankUrl(bankId, `/directives/${encodeURIComponent(directiveId)}`),
      { method: "DELETE", headers: getDataplaneHeaders() }
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error("API error deleting directive:", errorText);
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: errorText || "Failed to delete directive",
          errorKey: "api.errors.directives.delete",
        }),
        { status: response.status }
      );
    }

    return NextResponse.json({ success: true }, { status: 200 });
  } catch (error) {
    console.error("Error deleting directive:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to delete directive",
        errorKey: "api.errors.directives.delete",
      }),
      { status: 500 }
    );
  }
}
