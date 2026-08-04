import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";

import {
  ACCESS_KEY_COOKIE,
  SESSION_MAX_AGE_SECONDS,
  createSessionToken,
  createTenantSessionToken,
  getSessionSecret,
  isTenantAuthEnabled,
  sessionCookieOptions,
} from "@/lib/auth/session";
import { DATAPLANE_URL } from "@/lib/hindsight-client";

export async function POST(request: NextRequest) {
  const accessKey = process.env.HINDSIGHT_CP_ACCESS_KEY;
  const tenantAuth = isTenantAuthEnabled();

  // If no auth mechanism is configured at all, return 503
  if (!accessKey && !tenantAuth) {
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

  // Shared-access-key mode (unchanged). Checked first so that a deployment with
  // both configured still honours the operator's own key.
  if (accessKey && providedKey && constantTimeCompare(providedKey, accessKey)) {
    return setSession(request, await createSessionToken(accessKey));
  }

  // Tenant mode: the key the user typed is their own dataplane API key. The API
  // is the only thing that can say whether it is valid, and which tenant it is,
  // so ask it rather than keeping a second copy of that knowledge here.
  if (tenantAuth && providedKey) {
    const secret = getSessionSecret();
    if (!secret) {
      // Without a secret we cannot encrypt the key into the session, and
      // storing it in the clear is not an acceptable fallback.
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "Session secret not configured",
          errorKey: "api.errors.auth.accessKeyNotConfigured",
        }),
        { status: 503 }
      );
    }

    if (await isValidDataplaneKey(providedKey)) {
      return setSession(request, await createTenantSessionToken(providedKey, secret));
    }
  }

  return NextResponse.json(
    localizeApiErrorPayload(request, {
      error: "Invalid access key",
      errorKey: "api.errors.auth.invalidAccessKey",
    }),
    { status: 401 }
  );
}

function setSession(request: NextRequest, token: string) {
  const response = NextResponse.json({ success: true });

  response.cookies.set({
    name: ACCESS_KEY_COOKIE,
    value: token,
    ...sessionCookieOptions(request),
    maxAge: SESSION_MAX_AGE_SECONDS,
  });

  return response;
}

/**
 * Ask the dataplane whether this key authenticates. Any 2xx means the API
 * accepted it and resolved it to a tenant; 401/403 means it did not. A
 * transport error is treated as "no" — failing closed is the only safe
 * direction for a login check.
 */
async function isValidDataplaneKey(apiKey: string): Promise<boolean> {
  try {
    const response = await fetch(`${DATAPLANE_URL}/v1/default/banks`, {
      headers: { Authorization: `Bearer ${apiKey}` },
      cache: "no-store",
    });
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Constant-time string comparison to prevent timing attacks.
 */
function constantTimeCompare(a: string, b: string): boolean {
  if (a.length !== b.length) {
    return false;
  }

  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }

  return result === 0;
}
