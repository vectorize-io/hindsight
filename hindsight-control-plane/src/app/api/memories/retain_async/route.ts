import { NextRequest, NextResponse } from "next/server";
import { sdk, lowLevelClient } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

export async function POST(request: NextRequest) {
  let body;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }
  const bankId = body.bank_id || body.agent_id;

  if (!bankId) {
    return NextResponse.json({ error: "bank_id is required" }, { status: 400 });
  }

  const { items } = body;

  const response = await sdk.retainMemories({
    client: lowLevelClient,
    path: { bank_id: bankId },
    body: { items, async: true },
  });
  return respondWithSdk(response, "Failed to batch retain async");
}
