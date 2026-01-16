import { NextResponse } from "next/server";
import { sdk, lowLevelClient } from "@/lib/hindsight-client";

export async function DELETE(
  request: Request,
  { params }: { params: Promise<{ bankId: string; modelId: string }> }
) {
  try {
    const { bankId, modelId } = await params;

    if (!bankId) {
      return NextResponse.json({ error: "bank_id is required" }, { status: 400 });
    }

    if (!modelId) {
      return NextResponse.json({ error: "model_id is required" }, { status: 400 });
    }

    const response = await sdk.deleteMentalModel({
      client: lowLevelClient,
      path: { bank_id: bankId, model_id: modelId },
    });

    if (response.error) {
      console.error("API error deleting mental model:", response.error);
      return NextResponse.json({ error: "Failed to delete mental model" }, { status: 500 });
    }

    return NextResponse.json(response.data, { status: 200 });
  } catch (error) {
    console.error("Error deleting mental model:", error);
    return NextResponse.json({ error: "Failed to delete mental model" }, { status: 500 });
  }
}
