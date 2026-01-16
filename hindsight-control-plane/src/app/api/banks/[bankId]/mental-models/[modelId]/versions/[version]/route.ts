import { NextResponse } from "next/server";

const DATAPLANE_URL = process.env.HINDSIGHT_CP_DATAPLANE_API_URL || "http://localhost:8888";

export async function GET(
  request: Request,
  { params }: { params: Promise<{ bankId: string; modelId: string; version: string }> }
) {
  try {
    const { bankId, modelId, version } = await params;

    if (!bankId) {
      return NextResponse.json({ error: "bank_id is required" }, { status: 400 });
    }

    if (!modelId) {
      return NextResponse.json({ error: "model_id is required" }, { status: 400 });
    }

    if (!version) {
      return NextResponse.json({ error: "version is required" }, { status: 400 });
    }

    const response = await fetch(
      `${DATAPLANE_URL}/v1/default/banks/${bankId}/mental-models/${modelId}/versions/${version}`,
      {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error("API error getting mental model version:", errorText);
      return NextResponse.json(
        { error: errorText || "Failed to get mental model version" },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error getting mental model version:", error);
    return NextResponse.json({ error: "Failed to get mental model version" }, { status: 500 });
  }
}
