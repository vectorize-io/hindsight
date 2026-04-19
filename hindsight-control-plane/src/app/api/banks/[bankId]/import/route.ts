import { NextRequest, NextResponse } from "next/server";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ bankId: string }> }
) {
  try {
    const { bankId } = await params;
    const body = await request.json();
    const dryRun = request.nextUrl.searchParams.get("dry_run") === "true";

    // Direct fetch since the SDK doesn't have this operation yet
    const url = dataplaneBankUrl(bankId, `/import${dryRun ? "?dry_run=true" : ""}`);
    const response = await fetch(url, {
      method: "POST",
      headers: getDataplaneHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(body),
    });

    const data = await response.json();
    if (!response.ok) {
      return NextResponse.json(data, { status: response.status });
    }

    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error importing bank template:", error);
    return NextResponse.json({ error: "Failed to import bank template" }, { status: 500 });
  }
}
