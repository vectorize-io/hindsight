import type { NextRequest } from "next/server";
import { afterEach, describe, expect, it, vi } from "vitest";

import { apiTargetBankId, isCrossSiteWrite } from "@/lib/auth/request-guard";

const ORIGINAL_ENV = { ...process.env };

afterEach(() => {
  process.env = { ...ORIGINAL_ENV };
  vi.restoreAllMocks();
});

function fakeRequest({
  method = "GET",
  headers = {},
  search = "",
}: {
  method?: string;
  headers?: Record<string, string>;
  search?: string;
} = {}): NextRequest {
  const h = new Headers(headers);
  return {
    method,
    headers: h,
    nextUrl: {
      host: "cp.example.com",
      searchParams: new URLSearchParams(search),
    },
  } as unknown as NextRequest;
}

describe("apiTargetBankId", () => {
  it("reads the bank id from a bank-scoped collection path (decoded)", () => {
    expect(apiTargetBankId("/api/banks/u2--notes", fakeRequest())).toBe("u2--notes");
    expect(apiTargetBankId("/api/stats/u2/memories-timeseries", fakeRequest())).toBe("u2");
    expect(apiTargetBankId("/api/operations/u2%2Fx", fakeRequest())).toBe("u2/x");
  });

  it("reads the bank id from a query param when the path has none", () => {
    expect(apiTargetBankId("/api/graph", fakeRequest({ search: "bank_id=u2--g" }))).toBe("u2--g");
    expect(apiTargetBankId("/api/recall", fakeRequest({ search: "agent_id=u2" }))).toBe("u2");
  });

  it("returns null when no bank id is addressed", () => {
    expect(apiTargetBankId("/api/banks", fakeRequest())).toBeNull();
    expect(apiTargetBankId("/api/version", fakeRequest())).toBeNull();
  });

  it("does not treat a non-bank collection path segment as a bank id", () => {
    // /api/chunks/<id> is not a bank-scoped collection; its scope is enforced
    // in-route from the response, not from the path here.
    expect(apiTargetBankId("/api/chunks/abc", fakeRequest())).toBeNull();
  });
});

describe("isCrossSiteWrite", () => {
  it("never blocks safe methods", () => {
    expect(isCrossSiteWrite(fakeRequest({ method: "GET" }))).toBe(false);
    expect(
      isCrossSiteWrite(fakeRequest({ method: "GET", headers: { origin: "https://evil.com" } }))
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
          headers: { origin: "https://public.example.com", "x-forwarded-host": "public.example.com" },
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
    process.env.HINDSIGHT_CP_FRAME_ANCESTORS = "https://tokengate.example.com";
    expect(
      isCrossSiteWrite(
        fakeRequest({ method: "POST", headers: { origin: "https://tokengate.example.com" } })
      )
    ).toBe(false);
  });

  it("ignores CSP keywords in the embed allowlist", () => {
    process.env.HINDSIGHT_CP_FRAME_ANCESTORS = "'self' https://tokengate.example.com";
    expect(
      isCrossSiteWrite(fakeRequest({ method: "POST", headers: { origin: "https://evil.com" } }))
    ).toBe(true);
  });

  it("falls back to Sec-Fetch-Site when Origin is absent", () => {
    expect(
      isCrossSiteWrite(
        fakeRequest({ method: "POST", headers: { "sec-fetch-site": "cross-site" } })
      )
    ).toBe(true);
    expect(
      isCrossSiteWrite(
        fakeRequest({ method: "POST", headers: { "sec-fetch-site": "same-origin" } })
      )
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
