import { NextRequest, NextResponse } from "next/server";
import { lowLevelClient, sdk } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

type Params = { params: Promise<{ bankId: string }> };

export async function GET(request: NextRequest, { params }: Params) {
  const { bankId } = await params;
  const response = await sdk.getKnowledgeBankConfig({
    client: lowLevelClient,
    path: { bank_id: bankId },
  });
  return respondWithSdk(response, "Failed to fetch knowledge bank config", { request });
}

export async function PATCH(request: NextRequest, { params }: Params) {
  const { bankId } = await params;
  const body = await request.json().catch(() => null);
  if (body === null) return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  const response = await sdk.updateKnowledgeBankConfig({
    client: lowLevelClient,
    path: { bank_id: bankId },
    body: { updates: body.updates },
  });
  return respondWithSdk(response, "Failed to update knowledge bank config", { request });
}

export async function DELETE(request: NextRequest, { params }: Params) {
  const { bankId } = await params;
  const response = await sdk.resetKnowledgeBankConfig({
    client: lowLevelClient,
    path: { bank_id: bankId },
  });
  return respondWithSdk(response, "Failed to reset knowledge bank config", { request });
}
