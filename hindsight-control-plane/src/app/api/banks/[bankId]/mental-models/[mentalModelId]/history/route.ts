import { NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

export async function GET(
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
      dataplaneBankUrl(bankId, `/mental-models/${encodeURIComponent(mentalModelId)}/history`),
      { method: "GET", headers: getDataplaneHeaders() }
    );

    if (!response.ok) {
      if (response.status === 404) {
        return NextResponse.json(
          localizeApiErrorPayload(request, {
            error: "Mental model not found",
            errorKey: "api.errors.mentalModels.notFound",
          }),
          { status: 404 }
        );
      }
      throw new Error(`API returned ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error fetching mental model history:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to fetch mental model history",
        errorKey: "api.errors.mentalModels.history",
      }),
      { status: 500 }
    );
  }
}
