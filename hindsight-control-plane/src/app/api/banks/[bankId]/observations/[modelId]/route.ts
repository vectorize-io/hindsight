import { NextResponse } from "next/server";

const DATAPLANE_URL = process.env.HINDSIGHT_CP_DATAPLANE_API_URL || "http://localhost:8888";

export async function GET(
  request: Request,
  { params }: { params: Promise<{ bankId: string; modelId: string }> }
) {
  try {
    const { bankId, modelId } = await params;

    if (!bankId || !modelId) {
      return NextResponse.json({ error: "bank_id and model_id are required" }, { status: 400 });
    }

    const response = await fetch(
      `${DATAPLANE_URL}/v1/default/banks/${bankId}/mental-models/${modelId}`,
      { method: "GET" }
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error("API error getting mental model:", errorText);
      return NextResponse.json(
        { error: "Failed to get mental model" },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error getting mental model:", error);
    return NextResponse.json({ error: "Failed to get mental model" }, { status: 500 });
  }
}
