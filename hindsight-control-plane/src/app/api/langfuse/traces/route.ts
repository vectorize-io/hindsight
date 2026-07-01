import { NextRequest, NextResponse } from "next/server";

/**
 * GET /api/langfuse/traces
 *
 * Proxies to the Langfuse backend API to list traces.
 * Uses LANGFUSE_BASE_URL / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY from env.
 *
 * Query params:
 *   page     - page number (default: 1)
 *   limit    - items per page (default: 20, max: 100)
 *   userId   - optional filter by user ID
 */
export async function GET(request: NextRequest) {
  const baseUrl = process.env.LANGFUSE_BASE_URL || "http://localhost:3002";
  const publicKey = process.env.LANGFUSE_PUBLIC_KEY || process.env.HINDSIGHT_CP_LANGFUSE_PUBLIC_KEY;
  const secretKey = process.env.LANGFUSE_SECRET_KEY || process.env.HINDSIGHT_CP_LANGFUSE_SECRET_KEY;

  // If Langfuse is not configured, return a helpful empty response
  if (!publicKey || !secretKey) {
    return NextResponse.json({
      data: [],
      meta: { page: 1, limit: 20, totalItems: 0, totalPages: 0 },
      notice: "Langfuse not configured — set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY",
    });
  }

  const { searchParams } = new URL(request.url);
  const page = parseInt(searchParams.get("page") || "1");
  const limit = Math.min(parseInt(searchParams.get("limit") || "20"), 100);
  const userId = searchParams.get("userId") || undefined;

  try {
    const params = new URLSearchParams({
      page: String(page),
      limit: String(limit),
    });
    if (userId) params.set("userId", userId);

    const credentials = Buffer.from(`${publicKey}:${secretKey}`).toString("base64");
    const res = await fetch(`${baseUrl}/api/public/traces?${params}`, {
      headers: {
        Authorization: `Basic ${credentials}`,
        "Content-Type": "application/json",
      },
      signal: AbortSignal.timeout(10_000),
    });

    if (!res.ok) {
      const errorBody = await res.text().catch(() => "");
      return NextResponse.json(
        {
          data: [],
          meta: { page, limit, totalItems: 0, totalPages: 0 },
          error: `Langfuse returned HTTP ${res.status}: ${errorBody || res.statusText}`,
        },
        { status: res.status },
      );
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json(
      {
        data: [],
        meta: { page, limit, totalItems: 0, totalPages: 0 },
        error: message,
      },
      { status: 502 },
    );
  }
}
