import { NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

export async function PATCH(
  request: Request,
  { params }: { params: Promise<{ bankId: string; alias: string }> }
) {
  try {
    const { bankId, alias } = await params;

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

    const response = await fetch(
      dataplaneBankUrl(bankId, `/aliases/${encodeURIComponent(alias)}`),
      {
        method: "PATCH",
        headers: getDataplaneHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(body),
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error("API error setting primary bank alias:", errorText);
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "Failed to update bank alias",
          errorKey: "api.errors.bankAliases.update",
        }),
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error setting primary bank alias:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to update bank alias",
        errorKey: "api.errors.bankAliases.update",
      }),
      { status: 500 }
    );
  }
}

export async function DELETE(
  request: Request,
  { params }: { params: Promise<{ bankId: string; alias: string }> }
) {
  try {
    const { bankId, alias } = await params;

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
      dataplaneBankUrl(bankId, `/aliases/${encodeURIComponent(alias)}`),
      { method: "DELETE", headers: getDataplaneHeaders() }
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error("API error deleting bank alias:", errorText);
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "Failed to delete bank alias",
          errorKey: "api.errors.bankAliases.delete",
        }),
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error deleting bank alias:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to delete bank alias",
        errorKey: "api.errors.bankAliases.delete",
      }),
      { status: 500 }
    );
  }
}
