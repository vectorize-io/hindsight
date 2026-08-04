import type { NextRequest } from "next/server";

export const ACCESS_KEY_COOKIE = "hindsight_cp_access";
export const SESSION_MAX_AGE_SECONDS = 60 * 60 * 24;

const CLOCK_SKEW_TOLERANCE_SECONDS = 60;

/**
 * Tenant-key auth: instead of one shared access key, each person logs in with
 * their own dataplane API key, and the control plane makes every downstream
 * call as that key. Because the API resolves a key to its own tenant (schema),
 * one control plane can serve many users, each seeing only their own banks.
 *
 * Off by default — an unset HINDSIGHT_CP_TENANT_AUTH leaves the shared
 * access-key behaviour exactly as it was.
 */
export function isTenantAuthEnabled(): boolean {
  return process.env.HINDSIGHT_CP_TENANT_AUTH === "true";
}

const TENANT_TOKEN_PREFIX = "t1";

/**
 * Tenant session token: `t1.<issuedAtSeconds>.<base64url iv>.<base64url ciphertext>`.
 *
 * The user's API key is ENCRYPTED (AES-256-GCM) rather than merely signed. An
 * httpOnly cookie already keeps it away from page scripts, but encrypting means
 * a cookie lifted off disk cannot be unwrapped back into a working API key and
 * replayed straight against the REST/MCP endpoints — the blast radius stays the
 * UI session. `issuedAt` is bound in as additional authenticated data, so it
 * cannot be edited to extend the session's life.
 *
 * Rotating HINDSIGHT_CP_SESSION_SECRET invalidates every outstanding session.
 */
export function getSessionSecret(): string | undefined {
  return process.env.HINDSIGHT_CP_SESSION_SECRET || undefined;
}

async function aesKey(secret: string): Promise<CryptoKey> {
  // AES-GCM needs exactly 256 bits; the secret is arbitrary-length text.
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(secret));
  return crypto.subtle.importKey("raw", digest, { name: "AES-GCM" }, false, [
    "encrypt",
    "decrypt",
  ]);
}

export async function createTenantSessionToken(apiKey: string, secret: string): Promise<string> {
  const issuedAt = Math.floor(Date.now() / 1000).toString();
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const ciphertext = await crypto.subtle.encrypt(
    { name: "AES-GCM", iv, additionalData: new TextEncoder().encode(issuedAt) },
    await aesKey(secret),
    new TextEncoder().encode(apiKey)
  );
  return [
    TENANT_TOKEN_PREFIX,
    issuedAt,
    base64UrlEncode(iv),
    base64UrlEncode(new Uint8Array(ciphertext)),
  ].join(".");
}

/**
 * Returns the API key carried by a valid, unexpired tenant session, else null.
 *
 * Note this does NOT re-check the key against the API — that would add a round
 * trip to every request. Revocation still takes effect immediately, because the
 * actual data call made with this key is what gets rejected.
 */
export async function readTenantSessionKey(
  token: string | undefined,
  secret: string | undefined
): Promise<string | null> {
  if (!token || !secret) return null;

  const parts = token.split(".");
  if (parts.length !== 4 || parts[0] !== TENANT_TOKEN_PREFIX) return null;
  const [, issuedAtRaw, ivRaw, ciphertextRaw] = parts;

  const issuedAt = Number(issuedAtRaw);
  if (!Number.isInteger(issuedAt) || issuedAt <= 0) return null;

  const nowSeconds = Math.floor(Date.now() / 1000);
  if (issuedAt > nowSeconds + CLOCK_SKEW_TOLERANCE_SECONDS) return null;
  if (nowSeconds - issuedAt > SESSION_MAX_AGE_SECONDS) return null;

  try {
    const plaintext = await crypto.subtle.decrypt(
      {
        name: "AES-GCM",
        iv: base64UrlDecode(ivRaw),
        additionalData: new TextEncoder().encode(issuedAtRaw),
      },
      await aesKey(secret),
      base64UrlDecode(ciphertextRaw)
    );
    const apiKey = new TextDecoder().decode(plaintext);
    return apiKey || null;
  } catch {
    // Wrong secret, tampered ciphertext, or edited issuedAt — all mean "no session".
    return null;
  }
}

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

// Backed by a plain ArrayBuffer (not ArrayBufferLike) so the result satisfies
// BufferSource where WebCrypto expects it.
function base64UrlDecode(value: string): Uint8Array<ArrayBuffer> {
  const base64 = value.replace(/-/g, "+").replace(/_/g, "/");
  const binary = atob(base64.padEnd(Math.ceil(base64.length / 4) * 4, "="));
  const bytes = new Uint8Array(new ArrayBuffer(binary.length));
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function constantTimeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return result === 0;
}
