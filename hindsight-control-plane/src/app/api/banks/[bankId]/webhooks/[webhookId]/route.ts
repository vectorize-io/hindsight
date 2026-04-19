import { NextResponse } from "next/server";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

export async function PATCH(
  request: Request,
  { params }: { params: Promise<{ bankId: string; webhookId: string }> }
) {
  const { bankId, webhookId } = await params;
  const body = await request.json();
  const res = await fetch(dataplaneBankUrl(bankId, `/webhooks/${encodeURIComponent(webhookId)}`), {
    method: "PATCH",
    headers: getDataplaneHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) return NextResponse.json({ error: data.detail || "Failed" }, { status: res.status });
  return NextResponse.json(data);
}

export async function DELETE(
  request: Request,
  { params }: { params: Promise<{ bankId: string; webhookId: string }> }
) {
  const { bankId, webhookId } = await params;
  const res = await fetch(dataplaneBankUrl(bankId, `/webhooks/${encodeURIComponent(webhookId)}`), {
    method: "DELETE",
    headers: getDataplaneHeaders({ "Content-Type": "application/json" }),
  });
  const data = await res.json();
  if (!res.ok) return NextResponse.json({ error: data.detail || "Failed" }, { status: res.status });
  return NextResponse.json(data);
}
