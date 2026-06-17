import { NextResponse } from "next/server";

import {
  createApiKey,
  getCurrentOrgContext,
  jsonError,
  listApiKeys,
} from "@/lib/supabase-org/store";

export async function GET(request: Request) {
  try {
    const context = await getCurrentOrgContext(request);
    return NextResponse.json(
      { api_keys: await listApiKeys(context.selectedOrgId) },
      { status: 200 }
    );
  } catch (error) {
    return jsonError(error instanceof Error ? error.message : "Failed to list API keys", 400);
  }
}

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as {
      name?: string;
      bank_ids?: string[] | null;
      allowed_operations?: string[] | null;
    };
    if (!body.name) return jsonError("name is required", 400);
    const context = await getCurrentOrgContext(request);
    const apiKey = await createApiKey(
      context,
      body.name,
      body.bank_ids ?? null,
      body.allowed_operations ?? ["retain", "recall", "reflect"]
    );
    return NextResponse.json({ api_key: apiKey }, { status: 201 });
  } catch (error) {
    return jsonError(error instanceof Error ? error.message : "Failed to create API key", 400);
  }
}
