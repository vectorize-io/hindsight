import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { NextRequest, NextResponse } from "next/server";
import { middleware } from "@/middleware";
import { ACCESS_KEY_COOKIE, createSessionToken } from "@/lib/auth/session";

// The real i18n middleware can't load under vitest; locale rewriting isn't what's tested.
vi.mock("next-intl/middleware", () => ({ default: () => () => NextResponse.next() }));

const KEY = "test-access-key";

async function request(path: string, withSession: boolean) {
  const req = new NextRequest(new URL(path, "http://localhost:9999"));
  if (withSession) req.cookies.set(ACCESS_KEY_COOKIE, await createSessionToken(KEY));
  return middleware(req);
}

describe("middleware session handling", () => {
  beforeEach(() => vi.stubEnv("HINDSIGHT_CP_ACCESS_KEY", KEY));
  afterEach(() => vi.unstubAllEnvs());

  it("marks its own 401 so the browser knows to log in again", async () => {
    const res = await request("/api/banks", false);
    expect(res.status).toBe(401);
    expect(res.headers.get("x-hindsight-session-required")).toBe("1");
  });

  it("sends a signed-in visit to /login on to its returnTo", async () => {
    const res = await request("/login?returnTo=%2Fbanks%2Fa", true);
    expect(res.status).toBe(307);
    expect(res.headers.get("location")).toBe("http://localhost:9999/banks/a");
  });

  it("never follows an off-site returnTo", async () => {
    const res = await request("/login?returnTo=%2F%2Fevil.com", true);
    expect(res.headers.get("location")).toBe("http://localhost:9999/dashboard");
  });

  it("still shows the login form without a valid session", async () => {
    const res = await request("/login", false);
    expect(res.headers.get("location")).toBeNull();
  });
});
