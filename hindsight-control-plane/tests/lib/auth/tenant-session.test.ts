import { afterEach, describe, expect, it, vi } from "vitest";

import {
  SESSION_MAX_AGE_SECONDS,
  createSessionToken,
  createTenantSessionToken,
  isTenantAuthEnabled,
  readTenantSessionKey,
} from "@/lib/auth/session";

const SECRET = "server-side-session-secret";
const API_KEY = "hsk_0123456789abcdef0123456789abcdef";

afterEach(() => {
  vi.useRealTimers();
  delete process.env.HINDSIGHT_CP_TENANT_AUTH;
});

describe("isTenantAuthEnabled", () => {
  it("is off unless explicitly set to true", () => {
    expect(isTenantAuthEnabled()).toBe(false);
    process.env.HINDSIGHT_CP_TENANT_AUTH = "1";
    expect(isTenantAuthEnabled()).toBe(false);
    process.env.HINDSIGHT_CP_TENANT_AUTH = "true";
    expect(isTenantAuthEnabled()).toBe(true);
  });
});

describe("tenant session token", () => {
  it("round-trips the API key", async () => {
    const token = await createTenantSessionToken(API_KEY, SECRET);
    await expect(readTenantSessionKey(token, SECRET)).resolves.toBe(API_KEY);
  });

  it("does not carry the key in a recoverable form", async () => {
    const token = await createTenantSessionToken(API_KEY, SECRET);
    expect(token).not.toContain(API_KEY);
    // ...and the raw key is not merely base64'd into the token either.
    expect(token).not.toContain(btoa(API_KEY).replace(/=+$/, ""));
  });

  it("rejects a token sealed with a different secret", async () => {
    const token = await createTenantSessionToken(API_KEY, SECRET);
    await expect(readTenantSessionKey(token, "rotated-secret")).resolves.toBeNull();
  });

  it("rejects an edited issuedAt, so a session cannot be extended", async () => {
    const token = await createTenantSessionToken(API_KEY, SECRET);
    const parts = token.split(".");
    parts[1] = String(Number(parts[1]) + 5);
    await expect(readTenantSessionKey(parts.join("."), SECRET)).resolves.toBeNull();
  });

  it("rejects tampered ciphertext", async () => {
    const token = await createTenantSessionToken(API_KEY, SECRET);
    const parts = token.split(".");
    parts[3] = parts[3].slice(0, -2) + (parts[3].slice(-2) === "AA" ? "BB" : "AA");
    await expect(readTenantSessionKey(parts.join("."), SECRET)).resolves.toBeNull();
  });

  it("expires after the session max age", async () => {
    const token = await createTenantSessionToken(API_KEY, SECRET);
    vi.useFakeTimers();
    vi.setSystemTime(Date.now() + (SESSION_MAX_AGE_SECONDS + 60) * 1000);
    await expect(readTenantSessionKey(token, SECRET)).resolves.toBeNull();
  });

  it("rejects a shared-access-key token, which carries no tenant identity", async () => {
    const legacy = await createSessionToken("some-access-key");
    await expect(readTenantSessionKey(legacy, SECRET)).resolves.toBeNull();
  });

  it("returns null when no secret is configured", async () => {
    const token = await createTenantSessionToken(API_KEY, SECRET);
    await expect(readTenantSessionKey(token, undefined)).resolves.toBeNull();
    await expect(readTenantSessionKey(undefined, SECRET)).resolves.toBeNull();
  });
});
