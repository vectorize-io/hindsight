import { NextRequest, NextResponse } from "next/server";

const LANGFUSE_BASE_URL = process.env.LANGFUSE_BASE_URL || "http://localhost:3002";

function getBasicAuth(): string {
  const pk = process.env.LANGFUSE_PUBLIC_KEY || "";
  const sk = process.env.LANGFUSE_SECRET_KEY || "";
  return Buffer.from(`${pk}:${sk}`).toString("base64");
}

async function langfuseFetch(path: string) {
  const url = `${LANGFUSE_BASE_URL}${path}`;
  const res = await fetch(url, {
    headers: {
      Authorization: `Basic ${getBasicAuth()}`,
      "Content-Type": "application/json",
    },
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) {
    throw new Error(`Langfuse ${res.status}: ${await res.text().catch(() => "")}`);
  }
  return res.json();
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const limit = searchParams.get("limit") || "50";
    const page = searchParams.get("page") || "1";

    const data = await langfuseFetch(`/api/public/models?limit=${limit}&page=${page}`);
    return NextResponse.json(data);
  } catch (error) {
    console.error("[Langfuse] models fetch error:", error);
    return NextResponse.json(
      {
        data: [],
        meta: { page: 1, limit: 50, totalItems: 0, totalPages: 0 },
        error: error instanceof Error ? error.message : String(error),
      },
      { status: 200 }
    );
  }
}
