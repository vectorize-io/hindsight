import { NextResponse } from "next/server";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

export async function GET(request: Request, { params }: { params: Promise<{ bankId: string }> }) {
  try {
    const { bankId } = await params;
    if (!bankId) {
      return NextResponse.json({ error: "bank_id is required" }, { status: 400 });
    }

    const { searchParams } = new URL(request.url);
    const query = searchParams.toString();

    const url = dataplaneBankUrl(bankId, `/audit-logs/stats${query ? `?${query}` : ""}`);
    const response = await fetch(url, {
      method: "GET",
      headers: getDataplaneHeaders(),
    });

    const data = await response.json();
    if (!response.ok) {
      return NextResponse.json(
        { error: data.detail || "Failed to get audit log stats" },
        { status: response.status }
      );
    }

    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error getting audit log stats:", error);
    return NextResponse.json({ error: "Failed to get audit log stats" }, { status: 500 });
  }
}
