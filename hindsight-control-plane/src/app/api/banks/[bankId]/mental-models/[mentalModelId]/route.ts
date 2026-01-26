import { NextResponse } from "next/server";

const DATAPLANE_URL = process.env.HINDSIGHT_CP_DATAPLANE_API_URL || "http://localhost:8888";

export async function GET(
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
      `${DATAPLANE_URL}/v1/default/banks/${bankId}/mental-models/${mentalModelId}`,
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

export async function PATCH(
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

    const body = await request.json();

    const response = await fetch(
      `${DATAPLANE_URL}/v1/default/banks/${bankId}/mental-models/${mentalModelId}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error("API error updating mental model:", errorText);
      return NextResponse.json(
        { error: errorText || "Failed to update mental model" },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error updating mental model:", error);
    return NextResponse.json({ error: "Failed to update mental model" }, { status: 500 });
  }
}

export async function DELETE(
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
      `${DATAPLANE_URL}/v1/default/banks/${bankId}/mental-models/${mentalModelId}`,
      { method: "DELETE" }
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error("API error deleting mental model:", errorText);
      return NextResponse.json(
        { error: errorText || "Failed to delete mental model" },
        { status: response.status }
      );
    }

    return NextResponse.json({ success: true }, { status: 200 });
  } catch (error) {
    console.error("Error deleting mental model:", error);
    return NextResponse.json({ error: "Failed to delete mental model" }, { status: 500 });
  }
}
