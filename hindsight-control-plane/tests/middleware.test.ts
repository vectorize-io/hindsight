import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { NextRequest, NextResponse } from "next/server";
import { unstable_doesMiddlewareMatch } from "next/experimental/testing/server";
import { config, middleware } from "@/middleware";
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

describe("middleware body limit", () => {
  afterEach(() => vi.unstubAllEnvs());
  const post = (bytes: number) =>
    middleware(
      new NextRequest(new URL("/api/knowledge-banks/docs/files/ingest", "http://localhost:9999"), {
        method: "POST",
        headers: { "content-length": String(bytes) },
      })
    );

  it("refuses a body over the limit with a 413 rather than letting Next.js truncate it", async () => {
    vi.stubEnv("HINDSIGHT_CP_MAX_UPLOAD_SIZE", "1mb");
    const res = await post(2 * 1024 * 1024);
    expect(res.status).toBe(413);
    expect(await res.json()).toMatchObject({
      error: expect.stringMatching(/too large/),
      limitBytes: 1024 * 1024,
    });
    expect((await post(1024 * 1024)).status).toBe(200);
  });
});

describe("middleware matcher", () => {
  const matches = (path: string) => unstable_doesMiddlewareMatch({ config, url: `http://localhost${path}` });

  it("runs on every API route, including ids that end like a static file", () => {
    for (const path of [
      "/api/banks",
      "/api/banks/x",
      "/api/banks/notes.json",
      "/api/stats/demo.txt",
      "/api/banks/x/aliases/a.js",
      "/api/banks/x/directives/d.xml",
    ]) {
      expect(matches(path), path).toBe(true);
    }
  });

  it("still skips static assets and Next.js internals", () => {
    for (const path of ["/logo.png", "/fonts/inter.woff2", "/_next/static/chunk.js", "/robots.txt"]) {
      expect(matches(path), path).toBe(false);
    }
  });

  it("still runs on pages, including bank ids with dots", () => {
    expect(matches("/banks/SX.Products.GovComply.Build")).toBe(true);
  });
});
