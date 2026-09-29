import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { hindsightClient } from "@/lib/hindsight-client";

// Time axes accepted by the dataplane's list_memories endpoint.
type TimeField = "created_at" | "updated_at" | "mentioned_at" | "occurred_start" | "occurred_end";
const TIME_FIELDS = new Set<string>([
  "created_at",
  "updated_at",
  "mentioned_at",
  "occurred_start",
  "occurred_end",
]);

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
    const stateParam = searchParams.get("state");
    const state = stateParam === "valid" || stateParam === "invalidated" ? stateParam : undefined;
    const documentId = searchParams.get("document_id") || undefined;
    const entityId = searchParams.get("entity_id") || undefined;
    // Validated against the dataplane's enum here so a typo is a dropped parameter
    // rather than a 422 surfacing as a generic 500 from the catch below.
    const timeFieldParam = searchParams.get("time_field");
    const timeField = TIME_FIELDS.has(timeFieldParam ?? "")
      ? (timeFieldParam as TimeField)
      : undefined;
    const startDate = searchParams.get("start_date") || undefined;
    const endDate = searchParams.get("end_date") || undefined;

    const response = await hindsightClient.listMemories(bankId, {
      limit,
      offset,
      type,
      q,
      consolidationState,
      state,
      documentId,
      entityId,
      timeField,
      startDate,
      endDate,
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
