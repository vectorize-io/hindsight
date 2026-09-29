import { NextResponse } from "next/server";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";

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
    dataplaneBankUrl(bankId, `/webhooks/${encodeURIComponent(webhookId)}/deliveries?${qs}`),
    {
      headers: getDataplaneHeaders({ "Content-Type": "application/json" }),
    }
  );
  const data = await res.json();
  if (!res.ok)
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: data.detail || "Failed to fetch webhook deliveries",
        errorKey: "api.errors.webhooks.deliveries",
      }),
      { status: res.status }
    );
  return NextResponse.json(data);
}
