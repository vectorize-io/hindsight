import { NextRequest, NextResponse } from "next/server";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ bankId: string; webhookId: string }> }
) {
  const tenant = request.nextUrl.searchParams.get("tenant");
  const { bankId, webhookId } = await params;
  const { searchParams } = new URL(request.url);
  const limit = searchParams.get("limit") || "50";
  const cursor = searchParams.get("cursor");
  const qs = new URLSearchParams({ limit });
  if (cursor) qs.set("cursor", cursor);
  const res = await fetch(
    dataplaneBankUrl(bankId, `/webhooks/${encodeURIComponent(webhookId)}/deliveries?${qs}`),
    {
      headers: getDataplaneHeaders(tenant, { "Content-Type": "application/json" }),
    }
  );
  const data = await res.json();
  if (!res.ok) return NextResponse.json({ error: data.detail || "Failed" }, { status: res.status });
  return NextResponse.json(data);
}
