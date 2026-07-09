import { NextResponse, type NextRequest } from "next/server";

import {
  ACCESS_KEY_COOKIE,
  SESSION_MAX_AGE_SECONDS,
  createSessionToken,
  sessionCookieOptions,
  verifySessionToken,
} from "@/lib/auth/session";
import type { ControlPlaneAuthProviderAdapter } from "@/lib/auth/provider";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";

export const accessKeyAuthProvider: ControlPlaneAuthProviderAdapter = {
  id: "access_key",
  expectedDataplaneAuthProfile: "disabled",
  logoutEnabled: true,

  validateConfig() {
    if (!process.env.HINDSIGHT_CP_ACCESS_KEY) {
      throw new Error("HINDSIGHT_CP_AUTH_PROVIDER=access_key requires HINDSIGHT_CP_ACCESS_KEY");
    }
  },

  async isAuthenticated(request) {
    const accessKey = process.env.HINDSIGHT_CP_ACCESS_KEY || "";
    return verifySessionToken(request.cookies.get(ACCESS_KEY_COOKIE)?.value, accessKey);
  },

  getDataplaneHeaders(_request, extra) {
    const headers: Record<string, string> = { ...extra };
    const apiKey = process.env.HINDSIGHT_CP_DATAPLANE_API_KEY || "";
    if (apiKey) headers.Authorization = `Bearer ${apiKey}`;
    return headers;
  },

  getLoginConfig() {
    return {
      provider: "access_key",
      fields: [
        {
          name: "key",
          type: "password",
          placeholder: "Access Key",
          autocomplete: "off",
          required: true,
        },
      ],
      submitLabel: "Sign in",
    };
  },

  async login(request: NextRequest) {
    const accessKey = process.env.HINDSIGHT_CP_ACCESS_KEY;
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
    if (!providedKey || !constantTimeCompare(providedKey, accessKey)) {
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
  },

  logout(response, request) {
    response.cookies.set({
      name: ACCESS_KEY_COOKIE,
      value: "",
      ...sessionCookieOptions(request),
      maxAge: 0,
    });
  },
};

function constantTimeCompare(a: string, b: string): boolean {
  if (a.length !== b.length) return false;

  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }

  return result === 0;
}
