import { NextRequest, NextResponse } from "next/server";

/**
 * GET /api/monitoring/llm-stats
 *
 * Proxies to Hindsight API LLM request stats.
 * Returns aggregate statistics for LLM request volume, latency, and errors.
 */
export async function GET(request: NextRequest) {
  const hindsightApiUrl = process.env.HINDSIGHT_CP_DATAPLANE_API_URL || "http://localhost:8888";
  const defaultBank = "opencode";

  try {
    const res = await fetch(
      `${hindsightApiUrl}/v1/default/banks/${defaultBank}/llm-requests/stats`,
      {
        signal: AbortSignal.timeout(5000),
      },
    );

    if (!res.ok) {
      return NextResponse.json(
        { error: `Hindsight returned HTTP ${res.status}` },
        { status: res.status },
      );
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: message, stats: null }, { status: 502 });
  }
}
