import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ agentId: string }> }
) {
  try {
    const { agentId } = await params;
    const search = new URLSearchParams();
    for (const key of ["start_date", "end_date"]) {
      const value = request.nextUrl.searchParams.get(key);
      if (value) search.set(key, value);
    }
    const upstream = await fetch(dataplaneBankUrl(agentId, `/stats/summary?${search}`), {
      headers: getDataplaneHeaders(),
    });
    const body = await upstream.json();
    return NextResponse.json(body, { status: upstream.status });
  } catch (error) {
    console.error("Error fetching stats summary:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to fetch stats summary",
        errorKey: "api.errors.stats.summary",
      }),
      { status: 500 }
    );
  }
}
