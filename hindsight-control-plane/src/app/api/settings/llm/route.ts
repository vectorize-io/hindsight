import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { lowLevelClient, sdk } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

export async function GET(request: NextRequest) {
  const response = await sdk.getServerLlmConfig({ client: lowLevelClient });
  return respondWithSdk(response, "Failed to fetch server LLM config", { request });
}

export async function PUT(request: NextRequest) {
  let body;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Invalid JSON body",
        errorKey: "api.errors.auth.invalidRequestBody",
      }),
      { status: 400 }
    );
  }
  const response = await sdk.updateServerLlmConfig({
    client: lowLevelClient,
    body,
  });
  return respondWithSdk(response, "Failed to update server LLM config", { request });
}

export async function DELETE(request: NextRequest) {
  const response = await sdk.resetServerLlmConfig({ client: lowLevelClient });
  return respondWithSdk(response, "Failed to clear server LLM config", { request });
}
