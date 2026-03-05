import { NextResponse } from "next/server";
import { DATAPLANE_URL, getDataplaneHeaders } from "@/lib/hindsight-client";

export async function GET(request: Request, { params }: { params: Promise<{ bankId: string }> }) {
  const { bankId } = await params;
  const res = await fetch(`${DATAPLANE_URL}/v1/default/banks/${bankId}/webhooks`, {
    headers: getDataplaneHeaders({ "Content-Type": "application/json" }),
  });
  const data = await res.json();
  if (!res.ok) return NextResponse.json({ error: data.detail || "Failed" }, { status: res.status });
  return NextResponse.json(data);
}

export async function POST(request: Request, { params }: { params: Promise<{ bankId: string }> }) {
  const { bankId } = await params;
  const body = await request.json();
  const res = await fetch(`${DATAPLANE_URL}/v1/default/banks/${bankId}/webhooks`, {
    method: "POST",
    headers: getDataplaneHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) return NextResponse.json({ error: data.detail || "Failed" }, { status: res.status });
  return NextResponse.json(data, { status: 201 });
}
