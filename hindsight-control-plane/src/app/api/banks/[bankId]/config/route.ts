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
    const url = dataplaneBankUrl(bankId, "/config");
    const response = await fetch(url, { headers: getDataplaneHeaders() });
    const data = await response.json();
    if (!response.ok) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: data.detail || "Failed to get config",
          errorKey: "api.errors.config.get",
        }),
        { status: response.status }
      );
    }
    return NextResponse.json(data);
  } catch (error) {
    console.error("Error getting bank config:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to get config",
        errorKey: "api.errors.config.get",
      }),
      { status: 500 }
    );
  }
}

export async function PATCH(request: Request, { params }: { params: Promise<{ bankId: string }> }) {
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
    const body = await request.json();
    const url = dataplaneBankUrl(bankId, "/config");
    const response = await fetch(url, {
      method: "PATCH",
      headers: getDataplaneHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(body),
    });
    const data = await response.json();
    if (!response.ok) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: data.detail || "Failed to update config",
          errorKey: "api.errors.config.update",
        }),
        { status: response.status }
      );
    }
    return NextResponse.json(data);
  } catch (error) {
    console.error("Error updating bank config:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to update config",
        errorKey: "api.errors.config.update",
      }),
      { status: 500 }
    );
  }
}
