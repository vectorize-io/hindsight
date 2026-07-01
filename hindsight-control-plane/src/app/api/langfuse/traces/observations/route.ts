import { NextRequest, NextResponse } from "next/server";

/**
 * GET /api/langfuse/traces/observations?traceId=xxx
 *
 * Proxies to the Langfuse backend API to list observations (spans) for a trace.
 * Uses LANGFUSE_BASE_URL / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY from env.
 */
export async function GET(request: NextRequest) {
  const baseUrl = process.env.LANGFUSE_BASE_URL || "http://localhost:3002";
  const publicKey = process.env.LANGFUSE_PUBLIC_KEY || process.env.HINDSIGHT_CP_LANGFUSE_PUBLIC_KEY;
  const secretKey = process.env.LANGFUSE_SECRET_KEY || process.env.HINDSIGHT_CP_LANGFUSE_SECRET_KEY;

  const { searchParams } = new URL(request.url);
  const traceId = searchParams.get("traceId");

  if (!traceId) {
    return NextResponse.json(
      { data: [], error: "traceId query parameter is required" },
      { status: 400 }
    );
  }

  if (!publicKey || !secretKey) {
    return NextResponse.json({
      data: [],
      notice: "Langfuse not configured",
    });
  }

  try {
    const credentials = Buffer.from(`${publicKey}:${secretKey}`).toString("base64");
    const res = await fetch(
      `${baseUrl}/api/public/observations?traceId=${encodeURIComponent(traceId)}`,
      {
        headers: {
          Authorization: `Basic ${credentials}`,
          "Content-Type": "application/json",
        },
        signal: AbortSignal.timeout(10_000),
      }
    );

    if (!res.ok) {
      const errorBody = await res.text().catch(() => "");
      return NextResponse.json(
        { data: [], error: `Langfuse returned HTTP ${res.status}: ${errorBody || res.statusText}` },
        { status: res.status }
      );
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ data: [], error: message }, { status: 502 });
  }
}
