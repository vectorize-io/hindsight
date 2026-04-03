import { NextRequest, NextResponse } from "next/server";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ bankId: string; mentalModelId: string }> }
) {
  try {
    const tenant = request.nextUrl.searchParams.get("tenant");
    const { bankId, mentalModelId } = await params;

    if (!bankId || !mentalModelId) {
      return NextResponse.json(
        { error: "bank_id and mental_model_id are required" },
        { status: 400 }
      );
    }

    const response = await fetch(
      dataplaneBankUrl(bankId, `/mental-models/${encodeURIComponent(mentalModelId)}`),
      { method: "GET", headers: getDataplaneHeaders(tenant) }
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
  request: NextRequest,
  { params }: { params: Promise<{ bankId: string; mentalModelId: string }> }
) {
  try {
    const tenant = request.nextUrl.searchParams.get("tenant");
    const { bankId, mentalModelId } = await params;

    if (!bankId || !mentalModelId) {
      return NextResponse.json(
        { error: "bank_id and mental_model_id are required" },
        { status: 400 }
      );
    }

    const body = await request.json();

    const response = await fetch(
      dataplaneBankUrl(bankId, `/mental-models/${encodeURIComponent(mentalModelId)}`),
      {
        method: "PATCH",
        headers: getDataplaneHeaders(tenant, { "Content-Type": "application/json" }),
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
  request: NextRequest,
  { params }: { params: Promise<{ bankId: string; mentalModelId: string }> }
) {
  try {
    const tenant = request.nextUrl.searchParams.get("tenant");
    const { bankId, mentalModelId } = await params;

    if (!bankId || !mentalModelId) {
      return NextResponse.json(
        { error: "bank_id and mental_model_id are required" },
        { status: 400 }
      );
    }

    const response = await fetch(
      dataplaneBankUrl(bankId, `/mental-models/${encodeURIComponent(mentalModelId)}`),
      { method: "DELETE", headers: getDataplaneHeaders(tenant) }
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
