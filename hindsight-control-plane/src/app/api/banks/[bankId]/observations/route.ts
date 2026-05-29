import { NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { sdk, lowLevelClient } from "@/lib/hindsight-client";

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

    // Note: tags filtering is not supported by the list_memories API endpoint
    const response = await sdk.listMemories({
      client: lowLevelClient,
      path: { bank_id: bankId },
      query: {
        type: "observation",
        limit: 1000,
      },
    });

    if (response.error) {
      console.error("API error listing observations:", response.error);
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "Failed to list observations",
          errorKey: "api.errors.observations.list",
        }),
        { status: 500 }
      );
    }

    // Transform list memories response to observations format
    const items = (response.data?.items || []).map((item) => ({
      id: item.id,
      bank_id: bankId,
      text: item.text,
      proof_count: (item as any).proof_count ?? 1,
      history: [],
      tags: item.tags || [],
      source_memory_ids: [],
      source_memories: [],
      created_at: item.date,
      updated_at: item.date,
    }));

    return NextResponse.json({ items }, { status: 200 });
  } catch (error) {
    console.error("Error listing observations:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to list observations",
        errorKey: "api.errors.observations.list",
      }),
      { status: 500 }
    );
  }
}

export async function DELETE(
  request: Request,
  { params }: { params: Promise<{ bankId: string }> }
) {
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

    const response = await sdk.clearObservations({
      client: lowLevelClient,
      path: { bank_id: bankId },
    });

    if (response.error) {
      console.error("API error clearing observations:", response.error);
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "Failed to clear observations",
          errorKey: "api.errors.observations.clear",
        }),
        { status: 500 }
      );
    }

    return NextResponse.json(response.data, { status: 200 });
  } catch (error) {
    console.error("Error clearing observations:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to clear observations",
        errorKey: "api.errors.observations.clear",
      }),
      { status: 500 }
    );
  }
}
