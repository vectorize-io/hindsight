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

    const response = await fetch(
      dataplaneBankUrl(
        bankId,
        `/mental-models/${encodeURIComponent(mentalModelId)}/dry-run-refresh`
      ),
      { method: "POST", headers: getDataplaneHeaders() }
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error("API error dry-running mental model refresh:", errorText);
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: errorText || "Failed to dry-run mental model refresh",
          errorKey: "api.errors.mentalModels.dryRunRefresh",
        }),
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error dry-running mental model refresh:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to dry-run mental model refresh",
        errorKey: "api.errors.mentalModels.dryRunRefresh",
      }),
      { status: 500 }
    );
  }
}
