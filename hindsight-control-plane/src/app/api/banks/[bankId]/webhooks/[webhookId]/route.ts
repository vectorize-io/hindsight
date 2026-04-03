import { NextRequest, NextResponse } from "next/server";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ bankId: string; webhookId: string }> }
) {
  const tenant = request.nextUrl.searchParams.get("tenant");
  const { bankId, webhookId } = await params;
  const body = await request.json();
  const res = await fetch(dataplaneBankUrl(bankId, `/webhooks/${encodeURIComponent(webhookId)}`), {
    method: "PATCH",
    headers: getDataplaneHeaders(tenant, { "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) return NextResponse.json({ error: data.detail || "Failed" }, { status: res.status });
  return NextResponse.json(data);
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ bankId: string; webhookId: string }> }
) {
  const tenant = request.nextUrl.searchParams.get("tenant");
  const { bankId, webhookId } = await params;
  const res = await fetch(dataplaneBankUrl(bankId, `/webhooks/${encodeURIComponent(webhookId)}`), {
    method: "DELETE",
    headers: getDataplaneHeaders(tenant, { "Content-Type": "application/json" }),
  });
  const data = await res.json();
  if (!res.ok) return NextResponse.json({ error: data.detail || "Failed" }, { status: res.status });
  return NextResponse.json(data);
}
