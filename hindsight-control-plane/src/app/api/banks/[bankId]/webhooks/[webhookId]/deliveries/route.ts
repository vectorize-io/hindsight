import { NextResponse } from "next/server";
import { DATAPLANE_URL, getDataplaneHeaders } from "@/lib/hindsight-client";

export async function GET(
  request: Request,
  { params }: { params: Promise<{ bankId: string; webhookId: string }> }
) {
  const { bankId, webhookId } = await params;
  const { searchParams } = new URL(request.url);
  const limit = searchParams.get("limit") || "50";
  const cursor = searchParams.get("cursor");
  const qs = new URLSearchParams({ limit });
  if (cursor) qs.set("cursor", cursor);
  const res = await fetch(
    `${DATAPLANE_URL}/v1/default/banks/${bankId}/webhooks/${webhookId}/deliveries?${qs}`,
    {
      headers: getDataplaneHeaders({ "Content-Type": "application/json" }),
    }
  );
  const data = await res.json();
  if (!res.ok) return NextResponse.json({ error: data.detail || "Failed" }, { status: res.status });
  return NextResponse.json(data);
}
