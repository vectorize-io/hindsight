import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";

import {
  ACCESS_KEY_COOKIE,
  SESSION_MAX_AGE_SECONDS,
  createSessionToken,
  sessionCookieOptions,
} from "@/lib/auth/session";

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

  const providedKey = body.key;

  // Constant-time comparison to prevent timing attacks
  const isValid = providedKey && constantTimeCompare(providedKey, accessKey);

  if (!isValid) {
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
    value: await createSessionToken(accessKey),
    ...sessionCookieOptions(request),
    maxAge: SESSION_MAX_AGE_SECONDS,
  });

  return response;
}

/**
 * Constant-time string comparison to prevent timing attacks.
 * SECURITY FIX (CWE-208): The previous implementation early-returned false
 * on length mismatch, leaking the secret's length via response timing.
 * This version always iterates over the longer string's length, preventing
 * length oracle attacks while remaining constant-time for value comparison.
 */
function constantTimeCompare(a: string, b: string): boolean {
  const maxLen = Math.max(a.length, b.length);
  // Length mismatch is an error, but we must not return early — that leaks
  // the secret's length. Instead, flag it and keep iterating.
  let result = a.length ^ b.length;
  for (let i = 0; i < maxLen; i++) {
    // Index out-of-bounds returns NaN → XOR yields non-zero → comparison fails,
    // and we iterate the full maxLen either way.
    result |= (a.charCodeAt(i) || 0) ^ (b.charCodeAt(i) || 0);
  }

  return result === 0;
}
