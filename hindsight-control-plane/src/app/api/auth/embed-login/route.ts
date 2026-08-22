import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";

import {
  ACCESS_KEY_COOKIE,
  SESSION_MAX_AGE_SECONDS,
  createSessionToken,
  sessionCookieOptions,
} from "@/lib/auth/session";
import { sanitizeReturnTo, withBasePath } from "@/lib/base-path";

const DEFAULT_RETURN_TO = "/dashboard";

/**
 * Cross-site auto-login for an embedding page. A hidden `<form target=iframe>`
 * POSTs the access key here (form-encoded or JSON); we force-recreate the
 * session cookie and 302 to the sanitized `returnTo` so the iframe lands
 * directly on the dashboard. The 302 + Set-Cookie is what makes a single
 * cross-site POST render the app.
 *
 * This endpoint is only reachable from an origin listed in
 * `HINDSIGHT_CP_FRAME_ANCESTORS` (middleware applies the Origin check to
 * `/api/auth/*` too), so a hostile page cannot silently swap a visitor's
 * session. The embedder is responsible for deciding who may load the iframe;
 * the access key must stay server-side and never reach the browser.
 */
export async function POST(request: NextRequest) {
  const accessKey = process.env.HINDSIGHT_CP_ACCESS_KEY;
  const contentType = request.headers.get("content-type") ?? "";
  const isJson = contentType.includes("application/json");

  if (!accessKey) {
    return isJson
      ? NextResponse.json(
          localizeApiErrorPayload(request, {
            error: "Access key not configured",
            errorKey: "api.errors.auth.accessKeyNotConfigured",
          }),
          { status: 503 }
        )
      : htmlError(503, "Access key not configured");
  }

  let token: string | undefined;
  let returnTo: string | null | undefined;

  if (isJson) {
    try {
      const body = (await request.json()) as { token?: string; returnTo?: string };
      token = body.token;
      returnTo = body.returnTo;
    } catch {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "Invalid request body",
          errorKey: "api.errors.auth.invalidRequestBody",
        }),
        { status: 400 }
      );
    }
  } else {
    const form = await request.formData();
    const rawToken = form.get("token");
    const rawReturnTo = form.get("returnTo");
    token = typeof rawToken === "string" ? rawToken : undefined;
    returnTo = typeof rawReturnTo === "string" ? rawReturnTo : undefined;
  }

  if (!token || !constantTimeEqual(token, accessKey)) {
    return isJson
      ? NextResponse.json(
          localizeApiErrorPayload(request, {
            error: "Invalid access key",
            errorKey: "api.errors.auth.invalidAccessKey",
          }),
          { status: 401 }
        )
      : htmlError(401, "Invalid access key");
  }

  // Relative (path-only) Location so the browser resolves it against the public
  // origin it actually loaded, not the internal upstream host (request.url is
  // e.g. 0.0.0.0:9999 behind a proxy). sanitizeReturnTo already forbids
  // off-origin targets, so a relative path can never become an open redirect.
  const target = withBasePath(sanitizeReturnTo(returnTo, DEFAULT_RETURN_TO));
  const response = new NextResponse(null, {
    status: 302,
    // no-store so the session-bearing 302 is never cached by a shared proxy and
    // replayed to another viewer.
    headers: { Location: target, "Cache-Control": "no-store" },
  });

  response.cookies.set({
    name: ACCESS_KEY_COOKIE,
    value: await createSessionToken(accessKey),
    ...sessionCookieOptions(request),
    maxAge: SESSION_MAX_AGE_SECONDS,
  });

  return response;
}

function htmlError(status: number, message: string): NextResponse {
  return new NextResponse(`<!doctype html><html><body><p>${message}</p></body></html>`, {
    status,
    headers: { "content-type": "text/html; charset=utf-8" },
  });
}

function constantTimeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return result === 0;
}
