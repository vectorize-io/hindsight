import { NextRequest, NextResponse } from "next/server";
import { DATAPLANE_URL, getDataplaneHeaders } from "@/lib/hindsight-client";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ bankId: string }> }
) {
  try {
    const { bankId } = await params;

    const url = `${DATAPLANE_URL}/v1/default/banks/${encodeURIComponent(bankId)}/export`;
    const response = await fetch(url, {
      headers: getDataplaneHeaders(),
    });

    const data = await response.json();
    if (!response.ok) {
      return NextResponse.json(data, { status: response.status });
    }

    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error exporting bank template:", error);
    return NextResponse.json({ error: "Failed to export bank template" }, { status: 500 });
  }
}
