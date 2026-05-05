import { NextResponse } from "next/server";
import { getDataplaneHeaders, dataplaneBankUrl } from "@/lib/hindsight-client";

export async function POST(request: Request, { params }: { params: Promise<{ bankId: string }> }) {
  try {
    const { bankId } = await params;

    if (!bankId) {
      return NextResponse.json({ error: "bank_id is required" }, { status: 400 });
    }

    const response = await fetch(dataplaneBankUrl(bankId, "/tasks/recover"), {
      method: "POST",
      headers: getDataplaneHeaders(),
    });

    if (!response.ok) {
      const error = await response.text();
      console.error("API error recovering stuck tasks:", error);
      return NextResponse.json({ error: "Failed to recover stuck tasks" }, { status: response.status });
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error recovering stuck tasks:", error);
    return NextResponse.json({ error: "Failed to recover stuck tasks" }, { status: 500 });
  }
}
