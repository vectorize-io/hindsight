import { NextRequest } from "next/server";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/hindsight-client", () => ({
  DATAPLANE_URL: "http://dataplane",
  getDataplaneHeaders: () => ({}),
}));

import { GET } from "@/app/api/files/download/route";

async function download(headers: Record<string, string>) {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("bytes", { headers })));
  return GET(
    new NextRequest("http://localhost/api/files/download?path=/v1/default/files/download/banks/b/exports/x/a.zip")
  );
}

describe("GET /api/files/download", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("keeps the dataplane's filename on a download", async () => {
    const res = await download({ "content-type": "application/zip", "content-disposition": 'attachment; filename="b.zip"' });
    expect(res.headers.get("content-disposition")).toBe('attachment; filename="b.zip"');
    expect(res.headers.get("content-security-policy")).toContain("sandbox");
  });

  it("turns an inline file into a download", async () => {
    const res = await download({ "content-type": "text/html", "content-disposition": 'inline; filename="x.html"' });
    expect(res.headers.get("content-disposition")).toBe('attachment; filename="x.html"');
    expect(res.headers.get("x-content-type-options")).toBe("nosniff");
  });

  it("names the download after the path when the dataplane gives no disposition", async () => {
    const res = await download({ "content-type": "image/svg+xml" });
    expect(res.headers.get("content-disposition")).toBe('attachment; filename="a.zip"');
  });
});
