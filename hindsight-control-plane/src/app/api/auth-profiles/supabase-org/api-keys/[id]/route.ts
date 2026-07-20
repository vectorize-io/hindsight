import { NextResponse } from "next/server";

import {
  type ApiKeyPermissionMode,
  getCurrentOrgContext,
  jsonError,
  revealApiKey,
  revokeApiKey,
  updateApiKeyPermissions,
} from "@/lib/supabase-org/store";
import {
  type ApiKeyOperationScopeRequest,
  resolveOperationScopes,
} from "@/lib/supabase-org/api-key-scopes";

export async function GET(request: Request, { params }: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await params;
    const context = await getCurrentOrgContext(request);
    return NextResponse.json({ api_key: await revealApiKey(context, id) }, { status: 200 });
  } catch (error) {
    return jsonError(error instanceof Error ? error.message : "Failed to reveal API key", 400);
  }
}

export async function DELETE(request: Request, { params }: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await params;
    const context = await getCurrentOrgContext(request);
    await revokeApiKey(context, id);
    return NextResponse.json({ success: true }, { status: 200 });
  } catch (error) {
    return jsonError(error instanceof Error ? error.message : "Failed to revoke API key", 400);
  }
}

export async function PATCH(request: Request, { params }: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await params;
    const body = (await request.json()) as {
      permission_mode?: ApiKeyPermissionMode;
      operation_scopes?: ApiKeyOperationScopeRequest[] | null;
    };
    const permissionMode = body.permission_mode ?? "scoped";
    if (permissionMode !== "scoped" && permissionMode !== "full_access") {
      return jsonError("Invalid API key permission mode", 400);
    }
    const context = await getCurrentOrgContext(request);
    const operationScopes =
      permissionMode === "scoped"
        ? await resolveOperationScopes(request, body.operation_scopes ?? null)
        : null;
    await updateApiKeyPermissions(context, id, permissionMode, operationScopes);
    return NextResponse.json({ success: true }, { status: 200 });
  } catch (error) {
    return jsonError(error instanceof Error ? error.message : "Failed to update API key", 400);
  }
}
