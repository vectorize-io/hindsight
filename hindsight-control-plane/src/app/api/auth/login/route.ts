import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";

import {
  ACCESS_KEY_COOKIE,
  SESSION_MAX_AGE_SECONDS,
  createSessionToken,
  sessionCookieOptions,
} from "@/lib/auth/session";
import { resolveToken } from "@/lib/auth/tokens";

export async function POST(request: NextRequest) {
  const accessKey = process.env.HINDSIGHT_CP_ACCESS_KEY;

  // If no access key is configured, return 503
  if (!accessKey) {
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Access key not configured",
        errorKey: "api.errors.auth.accessKeyNotConfigured",
      }),
      { status: 503 }
    );
  }

  let body: { key?: string };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Invalid request body",
        errorKey: "api.errors.auth.invalidRequestBody",
      }),
      { status: 400 }
    );
  }

  const resolved = resolveToken(body.key);

  if (!resolved) {
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Invalid access key",
        errorKey: "api.errors.auth.invalidAccessKey",
      }),
      { status: 401 }
    );
  }

  const response = NextResponse.json({ success: true });

  response.cookies.set({
    name: ACCESS_KEY_COOKIE,
    value: await createSessionToken(accessKey, resolved.prefix),
    ...sessionCookieOptions(request),
    maxAge: SESSION_MAX_AGE_SECONDS,
  });

  return response;
}
