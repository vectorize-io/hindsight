import { NextRequest, NextResponse } from "next/server";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ bankId: string; modelId: string }> }
) {
  try {
    const tenant = request.nextUrl.searchParams.get("tenant");
    const { bankId, modelId } = await params;

    if (!bankId || !modelId) {
      return NextResponse.json({ error: "bank_id and model_id are required" }, { status: 400 });
    }

    const response = await fetch(
      dataplaneBankUrl(bankId, `/memories/${encodeURIComponent(modelId)}`),
      { method: "GET", headers: getDataplaneHeaders(tenant) }
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error("API error getting observation:", errorText);
      return NextResponse.json({ error: "Failed to get observation" }, { status: response.status });
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error getting mental model:", error);
    return NextResponse.json({ error: "Failed to get mental model" }, { status: 500 });
  }
}
