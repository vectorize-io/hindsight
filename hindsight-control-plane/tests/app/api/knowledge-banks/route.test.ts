import type { NextRequest } from "next/server";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/hindsight-client", () => ({
  DATAPLANE_URL: "http://dataplane",
  getDataplaneHeaders: (extra?: Record<string, string>) => ({
    Authorization: "Bearer secret",
    ...extra,
  }),
}));

import { GET, POST } from "@/app/api/knowledge-banks/[[...path]]/route";

function req(url: string, init?: RequestInit): NextRequest {
  const request = new Request(url, init) as NextRequest & { nextUrl: URL };
  Object.defineProperty(request, "nextUrl", { value: new URL(url) });
  return request;
}

const ctx = (path: string[]) => ({ params: Promise.resolve({ path }) });

afterEach(() => vi.restoreAllMocks());

describe("/api/knowledge-banks proxy", () => {
  it("forwards path, query and the dataplane auth header", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response('{"documents":[],"total":0}', {
        headers: { "content-type": "application/json", "content-length": "5" },
      })
    );
    const res = await GET(
      req("http://cp/api/knowledge-banks/my%2Fbank/documents?limit=25"),
      ctx(["my/bank", "documents"])
    );
    const [url, init] = fetchSpy.mock.calls[0];
    expect(url).toBe("http://dataplane/v1/default/knowledge-banks/my%2Fbank/documents?limit=25");
    expect((init?.headers as Record<string, string>).Authorization).toBe("Bearer secret");
    expect(await res.json()).toEqual({ documents: [], total: 0 });
    // fetch decompresses, so the upstream length must not be passed on.
    expect(res.headers.get("content-length")).toBeNull();
  });

  it("refuses dot segments that would escape the prefix", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const res = await GET(req("http://cp/api/knowledge-banks/x"), ctx(["..", "banks"]));
    expect(res.status).toBe(400);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("streams a multipart upload with the browser's boundary", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response('{"document_ids":["d"]}', { status: 202 }));
    const form = new FormData();
    form.append("files", new File(["hello"], "a.md", { type: "text/markdown" }));
    const browserReq = new Request("http://cp/api/knowledge-banks/docs/files", {
      method: "POST",
      body: form,
    });
    const contentType = browserReq.headers.get("content-type")!;
    const res = await POST(
      req("http://cp/api/knowledge-banks/docs/files", {
        method: "POST",
        body: await browserReq.arrayBuffer(),
        headers: { "content-type": contentType },
      }),
      ctx(["docs", "files"])
    );
    const init = fetchSpy.mock.calls[0][1]!;
    expect((init.headers as Record<string, string>)["Content-Type"]).toBe(contentType);
    expect(contentType).toMatch(/^multipart\/form-data; boundary=/);
    expect(res.status).toBe(202);
  });

  it("serves a byte range of passage media the dataplane sent whole", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(new Uint8Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), {
        headers: { "content-type": "video/mp4" },
      })
    );
    const res = await GET(
      req("http://cp/api/knowledge-banks/docs/documents/d/passages/0/media", {
        headers: { range: "bytes=2-4" },
      }),
      ctx(["docs", "documents", "d", "passages", "0", "media"])
    );
    expect(res.status).toBe(206);
    expect(res.headers.get("content-range")).toBe("bytes 2-4/10");
    expect(res.headers.get("content-type")).toBe("video/mp4");
    expect(res.headers.get("x-content-type-options")).toBe("nosniff");
    expect([...new Uint8Array(await res.arrayBuffer())]).toEqual([2, 3, 4]);
  });

  it("answers 416 for a range past the end of the media", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(new Uint8Array(4)));
    const res = await GET(
      req("http://cp/x", { headers: { range: "bytes=10-" } }),
      ctx(["docs", "documents", "d", "passages", "0", "media"])
    );
    expect(res.status).toBe(416);
    expect(res.headers.get("content-range")).toBe("bytes */4");
  });

  it("passes upstream errors through with their status", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response('{"detail":"knowledge bank not found"}', {
        status: 404,
        headers: { "content-type": "application/json" },
      })
    );
    const res = await GET(req("http://cp/x"), ctx(["nope"]));
    expect(res.status).toBe(404);
    expect(await res.json()).toEqual({ detail: "knowledge bank not found" });
  });

  it("names an untyped original by its file name so the browser can preview it", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(new Uint8Array([1, 2, 3]), {
        headers: {
          "content-type": "application/octet-stream",
          "content-disposition": "inline; filename*=UTF-8''demo%20clip.MP4",
        },
      })
    );
    const res = await GET(
      req("http://cp/api/knowledge-banks/docs/documents/d/file"),
      ctx(["docs", "documents", "d", "file"])
    );
    expect(res.headers.get("content-type")).toBe("video/mp4");
    expect(res.headers.get("content-disposition")).toContain("inline");
  });

  it("sandboxes an uploaded original, but not a PDF, which the browser would then not show", async () => {
    for (const [type, sandboxed] of [
      ["text/html", true],
      ["application/pdf", false],
    ] as const) {
      vi.spyOn(globalThis, "fetch").mockResolvedValue(
        new Response(new Uint8Array([1]), { headers: { "content-type": type } })
      );
      const res = await GET(
        req("http://cp/api/knowledge-banks/docs/documents/d/file"),
        ctx(["docs", "documents", "d", "file"])
      );
      expect(res.headers.get("x-content-type-options")).toBe("nosniff");
      expect(res.headers.get("content-security-policy")).toBe(sandboxed ? "sandbox" : null);
    }
  });
});
