import type { NextRequest } from "next/server";

export const ACCESS_KEY_COOKIE = "hindsight_cp_access";
export const SESSION_MAX_AGE_SECONDS = 60 * 60 * 24;

const CLOCK_SKEW_TOLERANCE_SECONDS = 60;

export type SessionVerification = {
  valid: boolean;
  prefix: string;
};

/**
 * Session token format: `<issuedAtSeconds>.<prefixB64url>.<base64urlHmacSha256>`.
 *
 * `prefixB64url` is the base64url of the (possibly empty) bank prefix the
 * session is scoped to. The HMAC is computed over `issuedAt + "." + prefixB64url`
 * using the access key as the secret, so neither the timestamp nor the prefix
 * can be forged without the key, and rotating the key invalidates every
 * outstanding session. No server-side state needed.
 */
export async function createSessionToken(accessKey: string, prefix = ""): Promise<string> {
  const issuedAt = Math.floor(Date.now() / 1000).toString();
  const prefixB64 = base64UrlEncode(new TextEncoder().encode(prefix));
  const payload = `${issuedAt}.${prefixB64}`;
  const signature = await hmacSha256Base64Url(accessKey, payload);
  return `${payload}.${signature}`;
}

export async function verifySessionToken(
  token: string | undefined,
  accessKey: string
): Promise<SessionVerification> {
  const invalid: SessionVerification = { valid: false, prefix: "" };
  if (!token) return invalid;

  const parts = token.split(".");
  // A legacy 2-part token (`<issuedAt>.<sig>`) no longer verifies — the user
  // simply re-logs in. Only the 3-part shape is accepted.
  if (parts.length !== 3) return invalid;

  const [issuedAtRaw, prefixB64, providedSignature] = parts;
  if (!issuedAtRaw || !providedSignature) return invalid;

  const issuedAt = Number(issuedAtRaw);
  if (!Number.isInteger(issuedAt) || issuedAt <= 0) return invalid;

  const nowSeconds = Math.floor(Date.now() / 1000);
  if (issuedAt > nowSeconds + CLOCK_SKEW_TOLERANCE_SECONDS) return invalid;
  if (nowSeconds - issuedAt > SESSION_MAX_AGE_SECONDS) return invalid;

  const payload = `${issuedAtRaw}.${prefixB64}`;
  const expectedSignature = await hmacSha256Base64Url(accessKey, payload);
  if (!constantTimeEqual(expectedSignature, providedSignature)) return invalid;

  return { valid: true, prefix: base64UrlDecodeToString(prefixB64) };
}

/**
 * Convenience for callers (e.g. middleware) that need the session's bank prefix:
 * reads the session cookie, verifies it, and returns the prefix when valid,
 * else null.
 */
export async function getSessionPrefix(request: NextRequest): Promise<string | null> {
  const accessKey = process.env.HINDSIGHT_CP_ACCESS_KEY;
  if (!accessKey) return null;
  const token = request.cookies.get(ACCESS_KEY_COOKIE)?.value;
  const result = await verifySessionToken(token, accessKey);
  return result.valid ? result.prefix : null;
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

/**
 * Cookie attributes for the session. When `HINDSIGHT_CP_COOKIE_SAMESITE=none`
 * the cookie is emitted as `SameSite=None; Secure; Partitioned` so it survives
 * inside a cross-site iframe (CHIPS). Otherwise it stays `SameSite=Lax` with
 * `Secure` derived from the request protocol, keeping plain-http dev working.
 *
 * `partitioned` is not yet in the Next.js cookie option types, so the return is
 * loosely typed to carry it through to `cookies().set(...)`.
 */
export function sessionCookieOptions(request: NextRequest): {
  httpOnly: boolean;
  secure: boolean;
  sameSite: "lax" | "none";
  path: string;
  partitioned?: boolean;
} {
  if (process.env.HINDSIGHT_CP_COOKIE_SAMESITE === "none") {
    return {
      httpOnly: true,
      secure: true,
      sameSite: "none",
      path: "/",
      partitioned: true,
    };
  }

  return {
    httpOnly: true,
    secure: isSecureRequest(request),
    sameSite: "lax",
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

function base64UrlDecodeToString(value: string): string {
  const normalized = value.replace(/-/g, "+").replace(/_/g, "/");
  const padded = normalized.padEnd(Math.ceil(normalized.length / 4) * 4, "=");
  const binary = atob(padded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new TextDecoder().decode(bytes);
}

function constantTimeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return result === 0;
}
