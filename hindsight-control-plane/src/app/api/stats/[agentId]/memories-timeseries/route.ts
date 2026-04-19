import { NextRequest, NextResponse } from "next/server";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ agentId: string }> }
) {
  try {
    const { agentId } = await params;
    const period = request.nextUrl.searchParams.get("period") || "7d";
    const url = dataplaneBankUrl(
      agentId,
      `/stats/memories-timeseries?period=${encodeURIComponent(period)}`
    );
    const upstream = await fetch(url, { headers: getDataplaneHeaders() });
    const body = await upstream.json();
    return NextResponse.json(body, { status: upstream.status });
  } catch (error) {
    console.error("Error fetching memories timeseries:", error);
    return NextResponse.json({ error: "Failed to fetch memories timeseries" }, { status: 500 });
  }
}
