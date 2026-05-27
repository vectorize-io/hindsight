import { NextResponse } from "next/server";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";

export async function GET(request: Request, { params }: { params: Promise<{ bankId: string }> }) {
  const { bankId } = await params;
  const res = await fetch(dataplaneBankUrl(bankId, "/webhooks"), {
    headers: getDataplaneHeaders({ "Content-Type": "application/json" }),
  });
  const data = await res.json();
  if (!res.ok)
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: data.detail || "Failed",
        errorKey: "api.errors.generic.failed",
      }),
      { status: res.status }
    );
  return NextResponse.json(data);
}

export async function POST(request: Request, { params }: { params: Promise<{ bankId: string }> }) {
  const { bankId } = await params;
  const body = await request.json();
  const res = await fetch(dataplaneBankUrl(bankId, "/webhooks"), {
    method: "POST",
    headers: getDataplaneHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok)
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: data.detail || "Failed",
        errorKey: "api.errors.generic.failed",
      }),
      { status: res.status }
    );
  return NextResponse.json(data, { status: 201 });
}
