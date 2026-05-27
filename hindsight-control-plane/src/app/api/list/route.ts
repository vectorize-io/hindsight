import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { hindsightClient } from "@/lib/hindsight-client";

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const bankId = searchParams.get("bank_id") || searchParams.get("agent_id");

    if (!bankId) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "bank_id is required",
          errorKey: "api.errors.validation.bankIdRequired",
        }),
        { status: 400 }
      );
    }

    const limit = searchParams.get("limit") ? Number(searchParams.get("limit")) : undefined;
    const offset = searchParams.get("offset") ? Number(searchParams.get("offset")) : undefined;
    const type = searchParams.get("type") || searchParams.get("fact_type") || undefined;
    const q = searchParams.get("q") || undefined;
    const consolidationStateParam =
      searchParams.get("consolidation_state") || searchParams.get("consolidationState");
    const consolidationState =
      consolidationStateParam === "failed" ||
      consolidationStateParam === "pending" ||
      consolidationStateParam === "done"
        ? consolidationStateParam
        : undefined;

    const response = await hindsightClient.listMemories(bankId, {
      limit,
      offset,
      type,
      q,
      consolidationState,
    });

    return NextResponse.json(response, { status: 200 });
  } catch (error) {
    console.error("Error listing memory units:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to list memory units",
        errorKey: "api.errors.memories.list",
      }),
      { status: 500 }
    );
  }
}

// Note: Individual memory unit deletion is not yet supported by the API
// Use clearBankMemories to delete all memories for a bank instead
export async function DELETE(request: NextRequest) {
  return NextResponse.json(
    localizeApiErrorPayload(request, {
      error:
        "Individual memory unit deletion is not yet supported. Use clear all memories instead.",
      errorKey: "api.errors.generic.unsupportedIndividualMemoryDelete",
    }),
    { status: 501 } // Not Implemented
  );
}
