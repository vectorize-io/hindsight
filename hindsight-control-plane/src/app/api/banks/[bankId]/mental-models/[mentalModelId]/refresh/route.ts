import { NextResponse } from "next/server";

const DATAPLANE_URL = process.env.HINDSIGHT_CP_DATAPLANE_API_URL || "http://localhost:8888";

export async function POST(
  request: Request,
  { params }: { params: Promise<{ bankId: string; mentalModelId: string }> }
) {
  try {
    const { bankId, mentalModelId } = await params;

    if (!bankId || !mentalModelId) {
      return NextResponse.json(
        { error: "bank_id and mental_model_id are required" },
        { status: 400 }
      );
    }

    const response = await fetch(
      `${DATAPLANE_URL}/v1/default/banks/${bankId}/mental-models/${mentalModelId}/refresh`,
      { method: "POST" }
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error("API error refreshing mental model:", errorText);
      return NextResponse.json(
        { error: errorText || "Failed to refresh mental model" },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error refreshing mental model:", error);
    return NextResponse.json({ error: "Failed to refresh mental model" }, { status: 500 });
  }
}
