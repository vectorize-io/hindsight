import type { NextRequest } from "next/server";
import { afterEach, describe, expect, it } from "vitest";

import { isCrossSiteWrite } from "@/lib/auth/request-guard";

const ORIGINAL_ENV = { ...process.env };

afterEach(() => {
  process.env = { ...ORIGINAL_ENV };
});

function fakeRequest({
  method = "GET",
  headers = {},
}: {
  method?: string;
  headers?: Record<string, string>;
} = {}): NextRequest {
  return {
    method,
    headers: new Headers(headers),
    nextUrl: { host: "cp.example.com" },
  } as unknown as NextRequest;
}

describe("isCrossSiteWrite", () => {
  it("never blocks safe methods, whatever the origin", () => {
    expect(isCrossSiteWrite(fakeRequest({ method: "GET" }))).toBe(false);
    expect(
      isCrossSiteWrite(fakeRequest({ method: "GET", headers: { origin: "https://evil.com" } }))
    ).toBe(false);
    expect(
      isCrossSiteWrite(fakeRequest({ method: "HEAD", headers: { origin: "https://evil.com" } }))
    ).toBe(false);
  });

  it("allows a same-origin write", () => {
    expect(
      isCrossSiteWrite(
        fakeRequest({ method: "POST", headers: { origin: "https://cp.example.com" } })
      )
    ).toBe(false);
  });

  it("honors x-forwarded-host for same-origin behind a proxy", () => {
    expect(
      isCrossSiteWrite(
        fakeRequest({
          method: "POST",
          headers: {
            origin: "https://public.example.com",
            "x-forwarded-host": "public.example.com",
          },
        })
      )
    ).toBe(false);
  });

  it("blocks a cross-site write from a non-allowlisted origin", () => {
    expect(
      isCrossSiteWrite(fakeRequest({ method: "POST", headers: { origin: "https://evil.com" } }))
    ).toBe(true);
  });

  it("allows a write from a configured embed origin", () => {
    process.env.HINDSIGHT_CP_FRAME_ANCESTORS = "https://app.example.com";
    expect(
      isCrossSiteWrite(
        fakeRequest({ method: "POST", headers: { origin: "https://app.example.com" } })
      )
    ).toBe(false);
  });

  it("ignores CSP keywords when reading the embed allowlist", () => {
    process.env.HINDSIGHT_CP_FRAME_ANCESTORS = "'self' https://app.example.com";
    expect(
      isCrossSiteWrite(fakeRequest({ method: "POST", headers: { origin: "https://evil.com" } }))
    ).toBe(true);
    expect(
      isCrossSiteWrite(
        fakeRequest({ method: "POST", headers: { origin: "https://app.example.com" } })
      )
    ).toBe(false);
  });

  it("does not treat a scheme/port mismatch as the same origin", () => {
    process.env.HINDSIGHT_CP_FRAME_ANCESTORS = "https://app.example.com";
    expect(
      isCrossSiteWrite(
        fakeRequest({ method: "POST", headers: { origin: "http://app.example.com" } })
      )
    ).toBe(true);
  });

  it("falls back to Sec-Fetch-Site when Origin is absent", () => {
    expect(
      isCrossSiteWrite(fakeRequest({ method: "POST", headers: { "sec-fetch-site": "cross-site" } }))
    ).toBe(true);
    expect(
      isCrossSiteWrite(
        fakeRequest({ method: "POST", headers: { "sec-fetch-site": "same-origin" } })
      )
    ).toBe(false);
    expect(
      isCrossSiteWrite(fakeRequest({ method: "POST", headers: { "sec-fetch-site": "none" } }))
    ).toBe(false);
  });

  it("allows non-browser writes that carry neither Origin nor Sec-Fetch-Site", () => {
    expect(isCrossSiteWrite(fakeRequest({ method: "POST" }))).toBe(false);
  });

  it("blocks a write with an unparseable Origin", () => {
    expect(
      isCrossSiteWrite(fakeRequest({ method: "POST", headers: { origin: "not a url" } }))
    ).toBe(true);
  });
});
