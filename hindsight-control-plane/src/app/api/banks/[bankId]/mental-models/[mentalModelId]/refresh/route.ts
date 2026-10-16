import { NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

export async function POST(
  request: Request,
  { params }: { params: Promise<{ bankId: string; mentalModelId: string }> }
) {
  try {
    const { bankId, mentalModelId } = await params;

    if (!bankId || !mentalModelId) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "bank_id and mental_model_id are required",
          errorKey: "api.errors.validation.bankAndMentalModelIdRequired",
        }),
        { status: 400 }
      );
    }

    // Optional body: { mode: "full" } for a one-off full refresh. A bodiless POST
    // (every caller before it existed) still refreshes in the model's own mode.
    const body = await request.json().catch(() => null);
    const mode = body?.mode === "full" ? "full" : undefined;
    const response = await fetch(
      dataplaneBankUrl(bankId, `/mental-models/${encodeURIComponent(mentalModelId)}/refresh`),
      mode
        ? {
            method: "POST",
            headers: getDataplaneHeaders({ "Content-Type": "application/json" }),
            body: JSON.stringify({ mode }),
          }
        : { method: "POST", headers: getDataplaneHeaders() }
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error("API error refreshing mental model:", errorText);
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: errorText || "Failed to refresh mental model",
          errorKey: "api.errors.mentalModels.refresh",
        }),
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error refreshing mental model:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to refresh mental model",
        errorKey: "api.errors.mentalModels.refresh",
      }),
      { status: 500 }
    );
  }
}
