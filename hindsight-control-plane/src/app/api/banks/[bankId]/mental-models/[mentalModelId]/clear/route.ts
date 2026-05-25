import { NextResponse } from "next/server";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

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
      dataplaneBankUrl(bankId, `/mental-models/${encodeURIComponent(mentalModelId)}/clear`),
      { method: "POST", headers: getDataplaneHeaders() }
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error("API error clearing mental model:", errorText);
      return NextResponse.json(
        { error: errorText || "Failed to clear mental model" },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error clearing mental model:", error);
    return NextResponse.json({ error: "Failed to clear mental model" }, { status: 500 });
  }
}
