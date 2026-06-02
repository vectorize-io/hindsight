import type { NextRequest } from "next/server";

export const ACCESS_KEY_COOKIE = "hindsight_cp_access";
export const SESSION_MAX_AGE_SECONDS = 60 * 60 * 24;

const CLOCK_SKEW_TOLERANCE_SECONDS = 60;

/**
 * Session token format: `<issuedAtSeconds>.<base64urlHmacSha256>`.
 *
 * The HMAC is computed over `issuedAtSeconds` using the access key as the
 * secret, so the token cannot be forged without knowing the key, and rotating
 * the key invalidates every outstanding session. No server-side state needed.
 */
export async function createSessionToken(accessKey: string): Promise<string> {
  const issuedAt = Math.floor(Date.now() / 1000).toString();
  const signature = await hmacSha256Base64Url(accessKey, issuedAt);
  return `${issuedAt}.${signature}`;
}

export async function verifySessionToken(
  token: string | undefined,
  accessKey: string
): Promise<boolean> {
  if (!token) return false;
  const separator = token.indexOf(".");
  if (separator <= 0 || separator === token.length - 1) return false;

  const payload = token.slice(0, separator);
  const providedSignature = token.slice(separator + 1);

  const issuedAt = Number(payload);
  if (!Number.isInteger(issuedAt) || issuedAt <= 0) return false;

  const nowSeconds = Math.floor(Date.now() / 1000);
  if (issuedAt > nowSeconds + CLOCK_SKEW_TOLERANCE_SECONDS) return false;
  if (nowSeconds - issuedAt > SESSION_MAX_AGE_SECONDS) return false;

  const expectedSignature = await hmacSha256Base64Url(accessKey, payload);
  return constantTimeEqual(expectedSignature, providedSignature);
}

/**
 * True when the original client connection used HTTPS. Honors
 * `X-Forwarded-Proto` from a TLS-terminating proxy first; falls back to the
 * request URL's protocol. We deliberately do NOT key off `NODE_ENV` — a
 * production build served over plain HTTP (common in self-hosted setups) must
 * still set a usable cookie.
 */
export function isSecureRequest(request: NextRequest): boolean {
  const forwardedProto = request.headers.get("x-forwarded-proto");
  if (forwardedProto) {
    return forwardedProto.split(",")[0]?.trim().toLowerCase() === "https";
  }
  return request.nextUrl.protocol === "https:";
}

export function sessionCookieOptions(request: NextRequest) {
  return {
    httpOnly: true,
    secure: isSecureRequest(request),
    sameSite: "lax" as const,
    path: "/",
  };
}

async function hmacSha256Base64Url(secret: string, message: string): Promise<string> {
  const encoder = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const signature = await crypto.subtle.sign("HMAC", key, encoder.encode(message));
  return base64UrlEncode(new Uint8Array(signature));
}

function base64UrlEncode(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/=+$/, "").replace(/\+/g, "-").replace(/\//g, "_");
}

function constantTimeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return result === 0;
}
