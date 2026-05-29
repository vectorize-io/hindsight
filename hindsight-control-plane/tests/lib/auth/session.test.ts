import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { NextRequest } from "next/server";

import {
  SESSION_MAX_AGE_SECONDS,
  createSessionToken,
  isSecureRequest,
  verifySessionToken,
} from "@/lib/auth/session";

const ACCESS_KEY = "super-secret-access-key";

function fakeRequest({
  protocol = "http:",
  forwardedProto,
}: {
  protocol?: "http:" | "https:";
  forwardedProto?: string;
} = {}): NextRequest {
  const headers = new Headers();
  if (forwardedProto !== undefined) {
    headers.set("x-forwarded-proto", forwardedProto);
  }
  return {
    headers,
    nextUrl: { protocol },
  } as unknown as NextRequest;
}

describe("createSessionToken / verifySessionToken", () => {
  it("round-trips: a freshly issued token verifies", async () => {
    const token = await createSessionToken(ACCESS_KEY);
    expect(await verifySessionToken(token, ACCESS_KEY)).toBe(true);
  });

  it("emits the documented `<issuedAt>.<sig>` shape", async () => {
    const token = await createSessionToken(ACCESS_KEY);
    const [issuedAt, sig, ...rest] = token.split(".");
    expect(rest).toHaveLength(0);
    expect(Number.isInteger(Number(issuedAt))).toBe(true);
    expect(sig.length).toBeGreaterThan(0);
  });

  it("rejects when the signature is tampered", async () => {
    const token = await createSessionToken(ACCESS_KEY);
    const [issuedAt] = token.split(".");
    const forged = `${issuedAt}.AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA`;
    expect(await verifySessionToken(forged, ACCESS_KEY)).toBe(false);
  });

  it("rejects when the payload is swapped (signature no longer matches)", async () => {
    const token = await createSessionToken(ACCESS_KEY);
    const [, sig] = token.split(".");
    const futureTime = (Math.floor(Date.now() / 1000) - 10).toString();
    const forged = `${futureTime}.${sig}`;
    expect(await verifySessionToken(forged, ACCESS_KEY)).toBe(false);
  });

  it("rejects the bare-string cookie value that the old impl accepted", async () => {
    expect(await verifySessionToken("authenticated", ACCESS_KEY)).toBe(false);
  });

  it.each(["", undefined, ".", "abc", "abc.", ".abc", "notanumber.sig"])(
    "rejects malformed token: %j",
    async (bad) => {
      expect(await verifySessionToken(bad, ACCESS_KEY)).toBe(false);
    }
  );

  it("rejects when the access key has rotated since the token was issued", async () => {
    const token = await createSessionToken(ACCESS_KEY);
    expect(await verifySessionToken(token, "different-access-key")).toBe(false);
  });

  describe("expiry", () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it("accepts a token issued just inside the max-age window", async () => {
      vi.setSystemTime(new Date("2026-01-01T00:00:00Z"));
      const token = await createSessionToken(ACCESS_KEY);
      vi.setSystemTime(new Date(Date.now() + (SESSION_MAX_AGE_SECONDS - 5) * 1000));
      expect(await verifySessionToken(token, ACCESS_KEY)).toBe(true);
    });

    it("rejects a token issued past the max-age window", async () => {
      vi.setSystemTime(new Date("2026-01-01T00:00:00Z"));
      const token = await createSessionToken(ACCESS_KEY);
      vi.setSystemTime(new Date(Date.now() + (SESSION_MAX_AGE_SECONDS + 5) * 1000));
      expect(await verifySessionToken(token, ACCESS_KEY)).toBe(false);
    });

    it("rejects a token whose issuedAt is implausibly in the future", async () => {
      vi.setSystemTime(new Date("2026-01-01T00:00:00Z"));
      const farFuture = Math.floor(Date.now() / 1000) + 3600;
      // Forge a properly-signed token with a future iat to isolate the iat
      // check (a tampered iat alone would also fail the signature check).
      const { hmacSha256 } = await loadHmacHelper();
      const futureSig = await hmacSha256(ACCESS_KEY, farFuture.toString());
      expect(await verifySessionToken(`${farFuture}.${futureSig}`, ACCESS_KEY)).toBe(false);
    });
  });
});

describe("isSecureRequest", () => {
  it("returns true when X-Forwarded-Proto is https", () => {
    expect(isSecureRequest(fakeRequest({ forwardedProto: "https" }))).toBe(true);
  });

  it("returns false when X-Forwarded-Proto is http (even on a production build)", () => {
    expect(isSecureRequest(fakeRequest({ forwardedProto: "http" }))).toBe(false);
  });

  it("uses the first value when X-Forwarded-Proto is a comma list", () => {
    expect(isSecureRequest(fakeRequest({ forwardedProto: "https, http" }))).toBe(true);
    expect(isSecureRequest(fakeRequest({ forwardedProto: "http, https" }))).toBe(false);
  });

  it("falls back to the request URL protocol when no forwarded header is set", () => {
    expect(isSecureRequest(fakeRequest({ protocol: "https:" }))).toBe(true);
    expect(isSecureRequest(fakeRequest({ protocol: "http:" }))).toBe(false);
  });

  it("is case-insensitive on the forwarded value", () => {
    expect(isSecureRequest(fakeRequest({ forwardedProto: "HTTPS" }))).toBe(true);
  });
});

// Inlined HMAC helper for the future-iat test — re-derives a signature without
// reaching into private module internals.
async function loadHmacHelper() {
  async function hmacSha256(secret: string, message: string): Promise<string> {
    const enc = new TextEncoder();
    const key = await crypto.subtle.importKey(
      "raw",
      enc.encode(secret),
      { name: "HMAC", hash: "SHA-256" },
      false,
      ["sign"]
    );
    const sig = new Uint8Array(await crypto.subtle.sign("HMAC", key, enc.encode(message)));
    let binary = "";
    for (const byte of sig) binary += String.fromCharCode(byte);
    return btoa(binary).replace(/=+$/, "").replace(/\+/g, "-").replace(/\//g, "_");
  }
  return { hmacSha256 };
}
